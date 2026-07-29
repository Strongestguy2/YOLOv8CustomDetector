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
    const width = sprite.natralWidth / dataset.sprite.columns;
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
    let light = 0; xLight = 0; yLight = 0;
    const quadrants = [0, 0, 0, 0];

    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
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

    const count = size * size, mx = xLight / light, my = yLight / light;
    let xVariance = 0, yVariance = 0, edges = 0;

    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
        const l = gray [y * size + x];
        xVariance += l * (((x + 0.5) / size) - mx) ** 2;
        yVariance += l * (((y + 0.5) / size) - my) ** 2;

        if (x + 1 < size) {
            edges += Math.abs (l - gray [y * size + x + 1]);
        }

        if (y + 1 < size) {
            edges += Math.abs (l - gray [(y + 1) * size + x]);
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
            deviations [i] = Math.sqrt (raw.reduce ((sum, row) => sum + (row [i] - mean [i]) ** 2, 0) / raw.length) || 1;
        }

        dataset.samples.forEach ((sample, index) => {
            sample.features = raw [index].map ((value, i) => i === 0 ? 1 : (value - means [i]) / deviations [i]);
        });
        featureCount = dimensions;
    }
}