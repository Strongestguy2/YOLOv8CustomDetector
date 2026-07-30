"use strict";

const ui = {
    train: document.querySelector ("#train-button"), stop : document.querySelector ("#stop-button"),
    status: document.querySelector ("#action-status"), badge : document.querySelector ("#run-badge"),
    epoch: document.querySelector ("#epoch-value"), loss : document.querySelector ("#loss-value"),
    sampleCount: document.querySelector ("#sample-value"), fill : document.querySelector ("#progress-fill"),
    progress: document.querySelector (".progress-track"), log : document.querySelector ("#log-body"),
    chart: document.querySelector ("#loss-chart"), preview : document.querySelector ("#preview"),
    sampleId: document.querySelector ("#sample-id")
};

let dataset, config, sprite;
let epoch = 0;
let running = false;
let previewIndex = 0;
let randomState = 8;
let featureCount = 0;
let weights = [];
let losses = [];
let logs = [];

const clamp = (value, low = 0.04, high = 0.96) => Math.min (high, Math.max (low, value));
const random = () => ((randomState = (randomState * 1664525 + 1013904223) >>> 0) / 4294967296);

function resetModel () {
    randomState = config.seed;
    weights = Array.from ({length : 5}, () => Array.from ({length : featureCount}, () => (random () - 0.5) * 0.08));
    epoch = 0;
    losses = [];
    logs = [];
}

function tileSource (sample) {
    const width = sprite.naturalWidth / dataset.sprite.columns;
    const height = sprite.naturalHeight / dataset.sprite.rows;
    return {x : sample.column * width, y : sample.row * height, width, height};
}

function rawImageFeatures (sample) {
    const size = config.image_size;
    const canvas = document.createElement ("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext ("2d", {willReadFrequently : true});
    const source = tileSource (sample);
    ctx.drawImage (sprite, source.x, source.y, source.width, source.height, 0, 0, size, size);
    const pixels = ctx.getImageData (0, 0, size, size).data;
    const gray = new Float32Array (size * size);
    let red = 0, green = 0, blue = 0;
    let light = 0, xLight = 0, yLight = 0;
    const quadrants = [0, 0, 0, 0];

    for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
            const pixel = y * size + x;
            const offset = pixel * 4;
            const r = pixels [offset] / 255;
            const g = pixels [offset + 1] / 255;
            const b = pixels [offset + 2] / 255;
            const l = 0.2126 * r + 0.7152 * g + 0.0722 * b + 0.001;

            gray [pixel] = l;
            red += r; green += g; blue += b;
            light += l; xLight += l * (x + 0.5) / size; yLight += l * (y + 0.5) / size;
            quadrants [(y >= size / 2 ? 2 : 0) + (x >= size / 2 ? 1 : 0)] += l;
        }
    }

    const count = size * size, mx = xLight / light, my = yLight / light;
    let xVariance = 0, yVariance = 0, edges = 0;

    for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
            const l = gray [y * size + x];
            xVariance += l * (((x + 0.5) / size) - mx) ** 2;
            yVariance += l * (((y + 0.5) / size) - my) ** 2;

            if (x + 1 < size) {
                edges += Math.abs (l - gray [y * size + x + 1]);
            }

            if (y + 1 < size) {
                edges += Math.abs (l - gray [(y + 1) * size + x]);
            }
        }
    }

    return [1, red / count, green / count, blue / count, mx, my,
        Math.sqrt (xVariance / light), Math.sqrt (yVariance / light), edges / (count * 2),
        ...quadrants.map (value => value / (count / 4))];
}

function prepareFeatures () {
    const raw = dataset.samples.map (rawImageFeatures);
    const dimensions = raw [0].length;
    const means = Array (dimensions).fill (0), deviations = Array (dimensions).fill (1);

    for (let i = 1; i < dimensions; i += 1) {
        means [i] = raw.reduce ((sum, row) => sum + row [i], 0) / raw.length;
        deviations [i] = Math.sqrt (raw.reduce ((sum, row) => sum + (row [i] - means [i]) ** 2, 0) / raw.length) || 1;
    }

    dataset.samples.forEach ((sample, index) => {
        sample.features = raw [index].map ((value, i) => i === 0 ? 1 : (value - means [i]) / deviations [i]);
    });
    featureCount = dimensions;
}

function predict (features) {
    const outputs = weights.map (row => row.reduce ((sum, value, i) => sum + value * features [i], 0));
    return [clamp (outputs [0]), clamp (outputs [1]), clamp (outputs [2], 0.08, 0.98), clamp (outputs [3], 0.08, 0.98), clamp (outputs [4], 0, 1)];
}

function trainEpoch () {
    const gradients = Array.from ({length : 5}, () => Array (featureCount).fill (0));
    let mse = 0;

    for (const sample of dataset.samples) {
        const target = [sample.cx, sample.cy, sample.width, sample.height, sample.class_id];
        const raw = weights.map (row => row.reduce ((sum, value, i) => sum + value * sample.features [i], 0));

        for (let output = 0; output < 5; output += 1) {
            const error = raw [output] - target [output];
            mse += error ** 2;

            for (let feature = 0; feature < featureCount; feature += 1) {
                gradients [output][feature] += 2 * error * sample.features [feature];
            }
        }
    }
        const scale = config.learning_rate / dataset.samples.length;

        for (let output = 0; output < 5; output += 1) {
            for (let feature = 0; feature < featureCount; feature += 1) {
                weights [output][feature] -= gradients [output][feature] * scale;
            }
        }

        return mse / (dataset.samples.length * 5);
}

function updateState (loss, state = "Training") {
    ui.epoch.textContent = `${epoch} / ${config.epochs}`;
    ui.loss.textContent = Number.isFinite (loss) ? loss.toFixed (5) : "-";
    ui.fill.style.width = `${(epoch / config.epochs) * 100}%`;
    ui.progress.setAttribute ("aria-valuenow", String (epoch));

    if (Number.isFinite (loss)) {
        logs.unshift ({epoch, loss, error : Math.sqrt (loss), lr : config.learning_rate, status : state});
        logs = logs.slice (0, 6);
    }

    ui.log.innerHTML = logs.length ? logs.map (row => `<tr><td>${row.epoch}</td><td>${row.loss.toFixed (6)}</td><td>${row.error.toFixed (5)}</td><td>${row.lr.toFixed (2)}</td><td>${row.status}</td></tr>`).join ("") : "<tr><td colspan='5' class='empty'>Click Train / Resume to begin.</td></tr>";
    drawChart ();
    drawPreview (dataset.samples [previewIndex % dataset.samples.length]);
}   

function drawChart () {
    const ctx = ui.chart.getContext ("2d");
    const {width, height} = ui.chart;
    
    ctx.clearRect (0, 0, width, height);
    ctx.strokeStyle = "#d2d2d7";
    ctx.lineWidth = 1;

    for (let i = 1; i < 5; i += 1) {
        const y = (height - 35) * i / 5;
        ctx.beginPath ();
        ctx.moveTo (45, y);
        ctx.lineTo (width - 16, y);
        ctx.stroke ();
    }

    ctx.fillStyle = "#6e6e73";
    ctx.font = "12px -apple-system, system-ui";
    ctx.fillText ("loss", 8, 17);
    ctx.fillText ("epoch", width - 48, height - 9);

    if (losses.length < 2) return;

    const maxLoss = Math.max (...losses, 0.01);
    
    ctx.strokeStyle = "#0071e3";
    ctx.lineWidth = 3;
    ctx.beginPath ();

    losses.forEach ((loss, index) => {
        const x = 45 + index / (config.epochs - 1) * (width - 65);
        const y = 15 + (1 - loss / maxLoss) * (height - 55);

        if (index === 0) ctx.moveTo (x, y); else ctx.lineTo (x, y);
    });

    ctx.stroke ();
}

function boxPixels (box, canvas) {
    return {x : (box [0] - box [2] / 2) * canvas.width, y : (box [1] - box [3] / 2) * canvas.height, w : box [2] * canvas.width, h : box [3] * canvas.height}
}

function drawPreview (sample) {
    const ctx = ui.preview.getContext ("2d");
    const source = tileSource (sample);
    const {width, height} = ui.preview;

    ctx.drawImage (sprite, source.x, source.y, source.width, source.height, 0, 0, width, height);

    const target = [sample.cx, sample.cy, sample.width, sample.height];
    const estimated = predict (sample.features);
    const targetBox = boxPixels (target, ui.preview);
    const predictedBox = boxPixels (estimated, ui.preview);

    ctx.strokeStyle = "#30d158";
    ctx.lineWidth = 4;
    ctx.strokeRect (targetBox.x, targetBox.y, targetBox.w, targetBox.h);

    ctx.strokeStyle = "#ff375f";
    ctx.lineWidth = 4;
    ctx.strokeRect (predictedBox.x, predictedBox.y, predictedBox.w, predictedBox.h);

    const predictedId = estimated [4] >= 0.5 ? 1 : 0;
    const confidence = predictedId ? estimated [4] : 1 - estimated [4];

    ctx.font = "600 17px -apple-system, system-ui";
    ctx.fillStyle = "rgba(29, 29, 31, .88)";
    ctx.fillRect (8, 8, 205, 29);
    ctx.fillStyle = "#fff";
    ctx.fillText  (`${dataset.class_names [predictedId]} ${(confidence * 100).toFixed (0)}%`, 16, 29);
    ui.sampleId.textContent = `${sample.id} · target: ${sample.class_name}`;
}

function finish (label, badgeClass, message) {
    running = false;
    ui.train.disabled = false;
    ui.stop.disabled = true;
    ui.badge.textContent = label;
    ui.badge.className = `badge ${badgeClass}`;
    ui.status.textContent = message;
}

function frame () {
    if (!running) return;

    for (let i = 0; i < 2 && epoch < config.epochs; i += 1) {
        losses.push (trainEpoch ());
        epoch += 1;
    }

    previewIndex = Math.floor (epoch / 10) % dataset.samples.length;
    updateState (losses.at (-1));
    ui.status.textContent = `Training cat/dog boxes... epoch ${epoch} / ${config.epochs}.`;
    
    if (epoch >= config.epochs) {
        finish ("COMPLETE", "ready", `Complete. Loss fell to ${losses.at (-1).toFixed (6)}. Click Train / Resume to replay.`)
    } else {
        window.setTimeout (() => window.requestAnimationFrame (frame), 35);
    }
}

function startTraining () {
    if (!dataset || running) return;
    if (epoch >= config.epochs) resetModel ();
    running = true;

    ui.train.disabled = true;
    ui.stop.disabled = false;
    ui.badge.textContent = "TRAINING";
    ui.badge.className = "badge training";
    frame ();
}

function stopTraining () {
    if (!running) return;
    finish ("STOPPED", "stopped", `Stop safely at epoch ${epoch}. Click Train / Resume to continue.`);
    updateState (losses.at (-1), "Stopped");
}

function loadImage (path) {
    return new Promise ((resolve, reject) => {
        const img = new Image ();
        img.onload = () => resolve (img);
        img.onerror = () => reject (new Error("photo sheet unavailable"));
        img.src = path;
    });
}

async function initialise () {
    try {
        [dataset, config] = await Promise.all ([
            fetch ("demo/dataset.json").then (response =>{ 
                if (!response.ok) throw new Error ("dataset unavailable"); 
                return response.json (); 
            }),
            fetch ("demo/config.json").then (response =>{ 
                if (!response.ok) throw new Error ("config unavailable"); 
                return response.json (); 
            })
        ]);

        sprite = await loadImage (dataset.sprite.path);
        prepareFeatures ();
        resetModel ();
        ui.sampleCount.textContent = String (dataset.samples.length);
        ui.status.textContent = "Ready. Click Train / Resume to learn the cat / dog classses and bounding boxes from the bundled photos.";
        updateState (Number.NaN);
    } catch (error) {
        ui.status.textContent = `Could not load the bundled demo: ${error.message}`;
        ui.train.disabled = true;
    }
}

ui.train.addEventListener ("click", startTraining);
ui.stop.addEventListener ("click", stopTraining);
initialise ();
