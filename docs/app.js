"use strict";

const ui = {
  train: document.querySelector("#train-button"),
  status: document.querySelector("#action-status"),
  badge: document.querySelector("#run-badge"),
  epoch: document.querySelector("#epoch-value"),
  loss: document.querySelector("#loss-value"),
  sampleCount: document.querySelector("#sample-value"),
  fill: document.querySelector("#progress-fill"),
  progress: document.querySelector(".progress-track"),
  log: document.querySelector("#log-body"),
  chart: document.querySelector("#loss-chart"),
  preview: document.querySelector("#preview"),
  sampleId: document.querySelector("#sample-id"),
  maxStepsRange: document.querySelector("#max-steps-range"),
  maxStepsNumber: document.querySelector("#max-steps-number"),
  maxHoursRange: document.querySelector("#max-hours-range"),
  maxHoursNumber: document.querySelector("#max-hours-number"),
  project: document.querySelector("#project-select"),
  importProject: document.querySelector("#import-project-panel")
};

let checkpointKey = "universal-yolo-tiny-pets-checkpoint-v3";
let dataset;
let config;
let sprite;
let epoch = 0;
let running = false;
let previewIndex = 0;
let randomState = 8;
let featureCount = 0;
let weights = [];
let losses = [];
let logs = [];
let targetSteps = 160;
let trainingStartedAt = 0;

const clamp = (value, low = 0.04, high = 0.96) => Math.min(high, Math.max(low, value));
const random = () => ((randomState = (randomState * 1664525 + 1013904223) >>> 0) / 4294967296);

function resetModel () {
  randomState = config.seed;
  weights = Array.from({length: 5}, () =>
    Array.from({length: featureCount}, () => (random() - 0.5) * 0.08)
  );
  epoch = 0;
  losses = [];
  logs = [];
  previewIndex = 0;
}

function saveCheckpoint () {
  try {
    localStorage.setItem(checkpointKey, JSON.stringify({epoch, weights, losses, logs, targetSteps}));
  } catch (_) {
    // Training still works when browser storage is unavailable.
  }
}

function restoreCheckpoint () {
  try {
    const saved = JSON.parse(localStorage.getItem(checkpointKey) || "null");
    const validWeights = Array.isArray(saved?.weights)
      && saved.weights.length === 5
      && saved.weights.every(row => Array.isArray(row) && row.length === featureCount);
    if (!validWeights) return false;
    if (Number.isFinite(Number(saved.targetSteps))) {
      targetSteps = Math.min(400, Math.max(20, Number(saved.targetSteps)));
      ui.maxStepsRange.value = String(targetSteps);
      ui.maxStepsNumber.value = String(targetSteps);
    }
    epoch = Math.max(0, Number(saved.epoch) || 0);
    weights = saved.weights;
    losses = Array.isArray(saved.losses) ? saved.losses.filter(Number.isFinite) : [];
    logs = Array.isArray(saved.logs) ? saved.logs.slice(0, 6) : [];
    previewIndex = Math.floor(epoch / 10) % dataset.samples.length;
    return epoch > 0;
  } catch (_) {
    return false;
  }
}

function tileSource (sample) {
  const width = sprite.naturalWidth / dataset.sprite.columns;
  const height = sprite.naturalHeight / dataset.sprite.rows;
  return {x: sample.column * width, y: sample.row * height, width, height};
}

function rawImageFeatures (sample) {
  const size = config.model.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", {willReadFrequently: true});
  const source = tileSource(sample);
  ctx.drawImage(sprite, source.x, source.y, source.width, source.height, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const gray = new Float32Array(size * size);
  let red = 0;
  let green = 0;
  let blue = 0;
  let light = 0;
  let xLight = 0;
  let yLight = 0;
  const quadrants = [0, 0, 0, 0];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x;
      const offset = pixel * 4;
      const r = pixels[offset] / 255;
      const g = pixels[offset + 1] / 255;
      const b = pixels[offset + 2] / 255;
      const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b + 0.001;
      gray[pixel] = luminance;
      red += r;
      green += g;
      blue += b;
      light += luminance;
      xLight += luminance * (x + 0.5) / size;
      yLight += luminance * (y + 0.5) / size;
      quadrants[(y >= size / 2 ? 2 : 0) + (x >= size / 2 ? 1 : 0)] += luminance;
    }
  }

  const count = size * size;
  const mx = xLight / light;
  const my = yLight / light;
  let xVariance = 0;
  let yVariance = 0;
  let edges = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const luminance = gray[y * size + x];
      xVariance += luminance * (((x + 0.5) / size) - mx) ** 2;
      yVariance += luminance * (((y + 0.5) / size) - my) ** 2;
      if (x + 1 < size) edges += Math.abs(luminance - gray[y * size + x + 1]);
      if (y + 1 < size) edges += Math.abs(luminance - gray[(y + 1) * size + x]);
    }
  }
  return [
    1, red / count, green / count, blue / count, mx, my,
    Math.sqrt(xVariance / light), Math.sqrt(yVariance / light), edges / (count * 2),
    ...quadrants.map(value => value / (count / 4))
  ];
}

function prepareFeatures () {
  const raw = dataset.samples.map(rawImageFeatures);
  const dimensions = raw[0].length;
  const means = Array(dimensions).fill(0);
  const deviations = Array(dimensions).fill(1);
  for (let i = 1; i < dimensions; i += 1) {
    means[i] = raw.reduce((sum, row) => sum + row[i], 0) / raw.length;
    deviations[i] = Math.sqrt(
      raw.reduce((sum, row) => sum + (row[i] - means[i]) ** 2, 0) / raw.length
    ) || 1;
  }
  dataset.samples.forEach((sample, index) => {
    sample.features = raw[index].map((value, i) => i === 0 ? 1 : (value - means[i]) / deviations[i]);
  });
  featureCount = dimensions;
}

function predict (features) {
  const outputs = weights.map(row => row.reduce((sum, value, i) => sum + value * features[i], 0));
  return [
    clamp(outputs[0]), clamp(outputs[1]), clamp(outputs[2], 0.08, 0.98),
    clamp(outputs[3], 0.08, 0.98), clamp(outputs[4], 0, 1)
  ];
}

function trainEpoch () {
  const gradients = Array.from({length: 5}, () => Array(featureCount).fill(0));
  let mse = 0;
  for (const sample of dataset.samples) {
    const target = [sample.cx, sample.cy, sample.width, sample.height, sample.class_id];
    const raw = weights.map(row => row.reduce((sum, value, i) => sum + value * sample.features[i], 0));
    for (let output = 0; output < 5; output += 1) {
      const error = raw[output] - target[output];
      mse += error * error;
      for (let feature = 0; feature < featureCount; feature += 1) {
        gradients[output][feature] += 2 * error * sample.features[feature];
      }
    }
  }
  const scale = config.train.learning_rate / dataset.samples.length;
  for (let output = 0; output < 5; output += 1) {
    for (let feature = 0; feature < featureCount; feature += 1) {
      weights[output][feature] -= scale * gradients[output][feature];
    }
  }
  return mse / (dataset.samples.length * 5);
}

function updateState (loss, state = "Training") {
  ui.epoch.textContent = `${epoch} / ${targetSteps}`;
  ui.loss.textContent = Number.isFinite(loss) ? loss.toFixed(5) : "-";
  ui.fill.style.width = `${Math.min(100, (epoch / targetSteps) * 100)}%`;
  ui.progress.setAttribute("aria-valuemax", String(targetSteps));
  ui.progress.setAttribute("aria-valuenow", String(epoch));
  if (Number.isFinite(loss)) {
    logs.unshift({epoch, loss, error: Math.sqrt(loss), lr: config.train.learning_rate, status: state});
    logs = logs.slice(0, 6);
  }
  ui.log.innerHTML = logs.length
    ? logs.map(row => `<tr><td>${row.epoch}</td><td>${Number(row.loss).toFixed(6)}</td><td>${Number(row.error).toFixed(5)}</td><td>${Number(row.lr).toFixed(3)}</td><td>${row.status}</td></tr>`).join("")
    : '<tr><td colspan="5" class="empty">Click Train / Resume to begin.</td></tr>';
  drawChart();
  drawPreview(dataset.samples[previewIndex % dataset.samples.length]);
}

function drawChart () {
  const ctx = ui.chart.getContext("2d");
  const {width, height} = ui.chart;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#ded8ce";
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const y = (height - 35) * i / 5;
    ctx.beginPath();
    ctx.moveTo(45, y);
    ctx.lineTo(width - 16, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#6b645b";
  ctx.font = "12px system-ui";
  ctx.fillText("loss", 8, 17);
  ctx.fillText("step", width - 42, height - 9);
  if (losses.length < 2) return;
  const maxLoss = Math.max(...losses, 0.01);
  ctx.strokeStyle = "#ff9d00";
  ctx.lineWidth = 3;
  ctx.beginPath();
  losses.forEach((loss, index) => {
    const x = 45 + index / Math.max(targetSteps - 1, 1) * (width - 65);
    const y = 15 + (1 - loss / maxLoss) * (height - 55);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function boxPixels (box, canvas) {
  return {
    x: (box[0] - box[2] / 2) * canvas.width,
    y: (box[1] - box[3] / 2) * canvas.height,
    w: box[2] * canvas.width,
    h: box[3] * canvas.height
  };
}

function drawPreview (sample) {
  const ctx = ui.preview.getContext("2d");
  const source = tileSource(sample);
  const {width, height} = ui.preview;
  ctx.drawImage(sprite, source.x, source.y, source.width, source.height, 0, 0, width, height);
  const targetBox = boxPixels([sample.cx, sample.cy, sample.width, sample.height], ui.preview);
  const estimated = predict(sample.features);
  const predictionBox = boxPixels(estimated, ui.preview);
  ctx.strokeStyle = "#ff9d00";
  ctx.lineWidth = 4;
  ctx.strokeRect(targetBox.x, targetBox.y, targetBox.w, targetBox.h);
  ctx.strokeStyle = "#c2410c";
  ctx.strokeRect(predictionBox.x, predictionBox.y, predictionBox.w, predictionBox.h);
  const predictedId = estimated[4] >= 0.5 ? 1 : 0;
  const confidence = predictedId ? estimated[4] : 1 - estimated[4];
  ctx.font = "600 17px system-ui";
  ctx.fillStyle = "#171717";
  ctx.fillRect(8, 8, 205, 29);
  ctx.fillStyle = "#fff";
  ctx.fillText(`${dataset.class_names[predictedId]} ${(confidence * 100).toFixed(0)}%`, 16, 29);
  ui.sampleId.textContent = `${sample.id} · target: ${sample.class_name}`;
}

function setIdleButton () {
  ui.train.textContent = "Train / Resume";
  ui.train.className = "primary";
}

function finish (label, badgeClass, message) {
  running = false;
  setIdleButton();
  ui.badge.textContent = label;
  ui.badge.className = `badge ${badgeClass}`;
  ui.status.textContent = message;
  saveCheckpoint();
}

function frame () {
  if (!running) return;
  const maxHours = Number(ui.maxHoursNumber.value) || 0;
  if (maxHours > 0 && performance.now() - trainingStartedAt >= maxHours * 60 * 60 * 1000) {
    finish("STOPPED", "stopped", `Time limit reached. Checkpoint saved at step ${epoch}.`);
    return;
  }
  for (let i = 0; i < 2 && epoch < targetSteps; i += 1) {
    losses.push(trainEpoch());
    epoch += 1;
  }
  previewIndex = Math.floor(epoch / 10) % dataset.samples.length;
  updateState(losses.at(-1));
  ui.status.textContent = `> Training Tiny Pets... step ${epoch} / ${targetSteps}`;
  if (epoch % 10 === 0) saveCheckpoint();
  if (epoch >= targetSteps) {
    finish("COMPLETE", "ready", `> Complete. Loss fell to ${losses.at(-1).toFixed(6)}.`);
  } else {
    window.setTimeout(() => window.requestAnimationFrame(frame), 35);
  }
}

function startTraining () {
  if (!dataset || running) return;
  targetSteps = Math.min(400, Math.max(20, Number(ui.maxStepsNumber.value) || config.train.total_steps));
  if (epoch >= targetSteps) resetModel();
  running = true;
  trainingStartedAt = performance.now();
  ui.train.textContent = "Stop training";
  ui.train.className = "stop";
  ui.badge.textContent = "TRAINING";
  ui.badge.className = "badge training";
  frame();
}

function stopTraining () {
  if (!running) return;
  finish("STOPPED", "stopped", `> Stopped safely at step ${epoch}. Click Train / Resume to continue.`);
  updateState(losses.at(-1), "Stopped");
}

function toggleTraining () {
  if (running) stopTraining(); else startTraining();
}

function syncPair (source, target, minimum, maximum) {
  const value = Math.min(maximum, Math.max(minimum, Number(source.value)));
  source.value = String(value);
  target.value = String(value);
  if (source === ui.maxStepsNumber || source === ui.maxStepsRange) {
    targetSteps = value;
    if (dataset) updateState(losses.at(-1));
  }
}

function setupTabs () {
  document.querySelectorAll(".tab-button").forEach(button => {
    button.addEventListener("click", () => {
      const tabName = button.dataset.tab;
      document.querySelectorAll(".tab-button").forEach(item => {
        const active = item === button;
        item.classList.toggle("active-tab", active);
        item.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        const active = panel.id === `${tabName}-tab`;
        panel.hidden = !active;
        panel.classList.toggle("active-panel", active);
      });
    });
  });
}

function loadImage (path) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("photo sheet unavailable"));
    image.src = path;
  });
}

async function initialise () {
  setupTabs();
  try {
    [dataset, config] = await Promise.all([
      fetch("demo/dataset.json").then(response => {
        if (!response.ok) throw new Error("dataset unavailable");
        return response.json();
      }),
      fetch("demo/config.json").then(response => {
        if (!response.ok) throw new Error("config unavailable");
        return response.json();
      })
    ]);
    checkpointKey = config.project.checkpoint_key || checkpointKey;
    sprite = await loadImage(dataset.sprite.path);
    prepareFeatures();
    resetModel();
    targetSteps = config.train.total_steps;
    ui.maxStepsRange.value = String(targetSteps);
    ui.maxStepsNumber.value = String(targetSteps);
    const restored = restoreCheckpoint();
    ui.sampleCount.textContent = String(dataset.samples.length);
    ui.status.textContent = restored
      ? `> Restored browser checkpoint at step ${epoch}. Click Train / Resume to continue.`
      : "> Ready. Click Train / Resume to learn cat/dog classes and bounding boxes.";
    updateState(losses.at(-1));
  } catch (error) {
    ui.status.textContent = `> Could not load the bundled demo: ${error.message}`;
    ui.train.disabled = true;
  }
}

ui.train.addEventListener("click", toggleTraining);
ui.project.addEventListener("change", () => {
  ui.importProject.hidden = ui.project.value !== "import";
});
ui.maxStepsRange.addEventListener("input", () => syncPair(ui.maxStepsRange, ui.maxStepsNumber, 20, 400));
ui.maxStepsNumber.addEventListener("change", () => syncPair(ui.maxStepsNumber, ui.maxStepsRange, 20, 400));
ui.maxHoursRange.addEventListener("input", () => syncPair(ui.maxHoursRange, ui.maxHoursNumber, 0, 4));
ui.maxHoursNumber.addEventListener("change", () => syncPair(ui.maxHoursNumber, ui.maxHoursRange, 0, 4));
initialise();
