const canvasHeight = 600;
const canvasWidth = 1200;
let c = document.querySelector(".canvas");
let startButton = document.querySelector(".start");
let inputField = document.querySelector("#input");
let hiddenField = document.querySelector("#hidden");
let outputField = document.querySelector("#output");
let learningRateField = document.querySelector("#learningRate");
let errorElement = document.querySelector(".error");
let ctx = c.getContext("2d");
ctx.lineWidth = 1;

let input = [0.2];
let hidden = [3];
let output = [0.4, 0.99];
let lr = 0.6;
let started = false;
let NN = null;

let NNdata = { radius: 0, centers: [], values: [] };

setInputs();
generateNN();

function generateNN() {
  if (NNdata.radius === 0) NN = createNN(input, hidden, output);
  let calculatedOutputs = (forwardResult = forwardPropagate(
    NN["layerSizes"],
    NN["weights"],
    NN["inputLayer"]
  ));
  generateNNUI(
    [input.length, ...hidden, output.length],
    calculatedOutputs,
    normalize(NN["weights"]),
    output
  );
  let error = getError(output, calculatedOutputs[calculatedOutputs.length - 1]);
  errorElement.innerHTML = "Error: " + error;
  NNdata.values = calculatedOutputs;
}

function activate(z) {
  return 1 / (1 + Math.exp(-z));
}

function createNN(inputLayer, hiddenLayerSizes, outputLayer) {
  layerSizes = [inputLayer.length, ...hiddenLayerSizes, outputLayer.length];
  weights = [];
  for (let i = 0; i < layerSizes.length - 1; i++) {
    layer1size = layerSizes[i];
    layer2size = layerSizes[i + 1];
    layerWeights = [];
    for (let j = 0; j < layer1size; j++) {
      neuronWeights = [];
      for (let k = 0; k < layer2size; k++) neuronWeights.push(Math.random());
      layerWeights.push(neuronWeights);
    }
    weights.push(layerWeights);
  }
  return { layerSizes: layerSizes, weights: weights, inputLayer: inputLayer };
}

function forwardPropagate(layerSizes, weights, inputLayer) {
  outputs = [inputLayer];
  for (let i = 0; i < layerSizes.length - 1; i++) {
    prevOutputs = outputs[i];
    currentOutputs = [];
    currentLayerSize = layerSizes[i + 1];
    for (let j = 0; j < currentLayerSize; j++) {
      currentOutput = 0;
      for (let k = 0; k < prevOutputs.length; k++)
        currentOutput += prevOutputs[k] * weights[i][k][j];
      currentOutputs.push(activate(currentOutput));
    }
    outputs.push(currentOutputs);
  }

  return outputs;
}

function getError(trueOutputs, predictedOutputs) {
  error = 0;
  for (let i = 0; i < trueOutputs.length; i++)
    error += 0.5 * Math.pow(trueOutputs[i] - predictedOutputs[i], 2);
  return error;
}

function getGradientDescent(
  layer,
  neuron1,
  neuron2,
  weights,
  outputs,
  predictedOutputs
) {
  finalResult = 0;
  if (layer == weights.length - 1) {
    result = outputs[outputs.length - 1][neuron2] - predictedOutputs[neuron2];
    result *=
      outputs[layer][neuron1] *
      (outputs[layer + 1][neuron2] * (1 - outputs[layer + 1][neuron2]));
    return result;
  }
  for (let ind = 0; ind < predictedOutputs.length; ind++) {
    result = outputs[outputs.length - 1][ind] - predictedOutputs[ind];
    result *=
      outputs[layer][neuron1] *
      (outputs[layer + 1][neuron2] * (1 - outputs[layer + 1][neuron2]));
    temp = neuron2;
    for (let i = layer + 1; i < outputs.length - 1; i++) {
      if (weights[i] && weights[i][neuron2] && weights[i][neuron2][ind]) {
        result *=
          weights[i][neuron2][ind] *
          (outputs[i + 1][ind] * (1 - outputs[i + 1][ind]));
      }
      neuron2 = ind;
    }
    neuron2 = temp;
    finalResult += result;
  }
  return finalResult;
}

function backPropagate(weights, learningRate, outputs, predictedOutputs) {
  for (let i = weights.length - 1; i > -1; i--) {
    for (let j = 0; j < weights[i].length; j++) {
      for (let k = 0; k < weights[i][j].length; k++) {
        gradientDescent = getGradientDescent(
          i,
          j,
          k,
          weights,
          outputs,
          predictedOutputs
        );
        weights[i][j][k] = weights[i][j][k] - learningRate * gradientDescent;
      }
    }
  }
}

function trainNN(
  inputLayer,
  hiddenLayerSizes,
  outputLayer,
  lr,
  finalError,
  maxIter = 10000
) {
  count = 0;
  error = 1;
  final = 0;
  while (error > finalError) {
    forwardResult = forwardPropagate(
      NN["layerSizes"],
      NN["weights"],
      NN["inputLayer"]
    );
    error = getError(outputLayer, forwardResult[forwardResult.length - 1]);
    backPropagate(NN["weights"], lr, forwardResult, outputLayer);
    count += 1;
    final = forwardResult;
    let clonedWeights = JSON.parse(JSON.stringify(NN["weights"]));
    ((final, weights, error) => {
      setTimeout(() => {
        generateNNUI(
          [inputLayer.length, ...hiddenLayerSizes, outputLayer.length],
          final,
          normalize(weights),
          output
        );
        errorElement.innerHTML = "Error: " + error;
        NNdata.values = final;
      }, count * 200);
    })(final, clonedWeights, error);
  }
  console.log(final);
}

function normalize(weights) {
  let min = 100000000;
  let max = 0;
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      for (let k = 0; k < weights[i][j].length; k++) {
        let wt = Math.abs(weights[i][j][k]);
        if (wt > max) max = wt;
        if (wt < min) min = wt;
      }
    }
  }
  let normalized = [];
  for (let i = 0; i < weights.length; i++) {
    let normalizedLayer = [];
    for (let j = 0; j < weights[i].length; j++) {
      let normalizedNeuron = [];
      for (let k = 0; k < weights[i][j].length; k++) {
        let wt = Math.abs(weights[i][j][k]);
        let normalizedWt = (wt - min) / (max - min);
        normalizedNeuron.push(normalizedWt);
      }
      normalizedLayer.push(normalizedNeuron);
    }
    normalized.push(normalizedLayer);
  }
  return normalized;
}

function generateLayerUI(n, outputs, x, radius) {
  let layer = [];
  const rad = canvasHeight / (3 * n + 1);
  let y = 2 * rad;
  for (let i = 0; i < n; i++) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    layer.push({ x: x, y: y });
    ctx.fillStyle = `rgba(255,255,255,${outputs[i]})`;
    ctx.fill();
    y += 3 * rad;
  }
  return layer;
}

function generateNNUI(layers, outputs, weights, output) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  let allCenters = [];
  let max = 0;
  for (let i = 0; i < layers.length; i++) {
    if (layers[i] > max) max = layers[i];
  }
  const rad = canvasWidth / (3 * layers.length + 4);
  let minRadius = canvasHeight / (3 * max + 1);
  if (rad < minRadius) minRadius = rad;
  let x = 2 * rad;
  NNdata.radius = minRadius;
  for (let i = 0; i < layers.length; i++) {
    let layerCenters = generateLayerUI(layers[i], outputs[i], x, minRadius);
    allCenters.push(layerCenters);
    x += 3 * rad;
  }
  NNdata.centers = allCenters;
  generateLayerUI(layers[layers.length - 1], output, x - rad, minRadius);

  for (let i = 0; i < allCenters.length - 1; i++) {
    let layer1 = allCenters[i];
    let layer2 = allCenters[i + 1];
    for (let j = 0; j < layer1.length; j++) {
      for (let k = 0; k < layer2.length; k++) {
        ctx.beginPath();
        ctx.moveTo(layer1[j].x, layer1[j].y);
        ctx.lineTo(layer2[k].x, layer2[k].y);
        ctx.strokeStyle = `rgba(102, 161, 180, ${weights[i][j][k]})`;
        ctx.stroke();
      }
    }
  }
}

startButton.addEventListener("click", () => {
  trainNN(input, hidden, output, lr, 0.0001);
  started = true;
});

inputField.addEventListener("input", (e) => {
  NNdata = { radius: 0, centers: [], values: [] };
  let temp = e.target.value.split(" ").map((u) => Number(u));
  if (temp.some((u) => u < 0 || u > 1)) {
    window.alert("INVALID INPUT");
    setInputs();
    return;
  }
  input = temp;
  generateNN();
});

hiddenField.addEventListener("input", (e) => {
  NNdata = { radius: 0, centers: [], values: [] };
  let temp = e.target.value.split(" ").map((u) => Number(u));
  if (temp.some((u) => u % 1 != 0)) {
    window.alert("INVALID INPUT");
    setInputs();
    return;
  }
  hidden = temp;
  generateNN();
});

outputField.addEventListener("input", (e) => {
  NNdata = { radius: 0, centers: [], values: [] };
  let temp = e.target.value.split(" ").map((u) => Number(u));
  if (temp.some((u) => u < 0 || u > 1)) {
    window.alert("INVALID INPUT");
    setInputs();
    return;
  }
  output = temp;
  generateNN();
});

learningRateField.addEventListener("input", (e) => {
  NNdata = { radius: 0, centers: [], values: [] };
  if (e.target.value > 1 || e.target.value < 0) {
    setInputs();
    return window.alert("INVALID INPUT");
  }
  lr = e.target.value;
});

c.addEventListener("mousemove", (e) => {
  let canvasX = e.pageX - 150;
  let canvasY = e.pageY - 150;
  for (let i = 0; i < NNdata.centers.length; i++) {
    for (let j = 0; j < NNdata.centers[i].length; j++) {
      let h = NNdata.centers[i][j].x;
      let k = NNdata.centers[i][j].y;
      let isInside =
        Math.sqrt(Math.pow(canvasX - h, 2) + Math.pow(canvasY - k, 2)) <=
        NNdata.radius;
      if (isInside) {
        ctx.fillStyle = "#de30de";
        ctx.font = "20px Arial";
        ctx.fillText(
          "Value: " + NNdata.values[i][j].toFixed(3),
          h + NNdata.radius / 2,
          k
        );
        return;
      }
    }
  }
  if (!started) generateNN();
});

function setInputs() {
  inputField.value = input.reduce((acc, curr) => (acc += curr + " "), "");
  hiddenField.value = hidden.reduce((acc, curr) => (acc += curr + " "), "");
  outputField.value = output.reduce((acc, curr) => (acc += curr + " "), "");
  learningRateField.value = lr;
}
