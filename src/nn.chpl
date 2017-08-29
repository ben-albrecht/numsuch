module NN {
  use LinearAlgebra;
  use Random;

  class Sequential {
    var layerDom = {1..0},
        layers: [layerDom] Layer;

    proc add(d: Dense) {
      writeln(" Adding a new dense layer");
      //var l = new Layer(units=d.units, inputDim=d.inputDim, layerId=layerDom.last);
      layers.push_back(new Layer());
      ref currentLayer = layers[layerDom.last];
      currentLayer.layerId = layerDom.last;
      if currentLayer.layerId == layerDom.first {
        currentLayer.inputDim = d.inputDim;
        currentLayer.batchSize = d.batchSize;
        currentLayer.weightDom = {1..#d.inputDim, 1..#d.batchSize};
      }
      currentLayer.units = d.units;
    }

    proc add(d: Activation) {
      writeln(" Adding a new activation layer");
      ref currentLayer = layers[layerDom.last];
      currentLayer.activation = new Activation(name=d.name);
    }

    //proc add(d: Activation) { }

    proc compile(xTrain:[], yTrain:[]) {
      for l in layers {
        if l.layerId == layerDom.first {
          /*
          writeln("This is the data layer");
          writeln(" current weightDom: ", l.weightDom);
          writeln("Input dimension is ", xTrain.shape);
          writeln("Label dimension is ", yTrain.shape);
           */
          l.W = xTrain.T;
        } else {
          ref lowerLayer = layers[l.layerId-1];
          l.inputDim = lowerLayer.units;
          l.batchSize = lowerLayer.batchSize;
          l.weightDom = {1..#l.units, 1..#l.inputDim};
          l.outputDom = {1..#l.units, 1..#l.batchSize};
        }
        l.predDom = {1..1, 1..#l.batchSize};
        //writeln(" currentLayer ", l);
      }
      var topLayer = new Layer(units=yTrain.shape[1]);
    }
    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
        compile(xTrain, yTrain);
        return MLPfit(xTrain, yTrain, epochs, lr);
    }
  }

  class Dense {
    var units: int,
        batchSize: int,
        inputDim: int;
  }
  class Activation {
    var name: string;
  }
  class Layer {
    var layerId: int,
    activation: Activation,
    batchSize: int,
    inputDim: int,
    units: int,
    weightDom: domain(2),  // units x inputDim
    outputDom: domain(2),  // units x batchSize = number of labels
    predDom: domain(2),
    //W: [{1..#units, 1..#inputDim}] real,
    W: [weightDom] real,
    a: [outputDom] real,
    h: [outputDom] real, // will be used in NEXT layer
    gradH: [outputDom]real, // the gradient of the output
    dH: [outputDom] real, // will be used in NEXT layer
    b: [outputDom]real, // The single column of bias
    bias: [outputDom] real, // bias = [b,b,..]
    error: [predDom] real,
    yHat: [predDom] real;
  }

  proc MLPfit(xTrain:[], yTrain:[], epochs:int, lr: real){
    var batchSize = xTrain.shape[1],
        inputDim = xTrain.shape[2],
        labelDim = yTrain.shape;

    for e in 1..#epochs {

    }

    return 0;
  }

}
