module NN {
  use LinearAlgebra,
      Time,
      Random;

  class Sequential {
    var layerDom = {1..0},
        L: Loss,
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
        currentLayer.weightDom = {1..#d.units, 1..#d.inputDim};
        currentLayer.outputDom = {1..#d.units, 1..#d.batchSize};
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
      //var X: [{1..1,1..1}] real;
      //X.domain = xTrain.domain;
      //X = xTrain;

      for l in layers {
        if l.layerId == layerDom.first {
          /*
          writeln("This is the data layer");
          writeln(" current weightDom: ", l.weightDom);
          writeln("Input dimension is ", xTrain.shape);
          writeln("Label dimension is ", yTrain.shape);
           */
          fillRandom(l.W);
          var b: [{l.bias.domain.dim(1)}] real;
          fillRandom(b);
          [j in l.bias.domain.dim(2)] l.bias[..,j] = b;
          //l.h = xTrain.T;
        } else {
          ref lowerLayer = layers[l.layerId-1];
          l.inputDim = lowerLayer.units;
          l.batchSize = lowerLayer.batchSize;
          l.weightDom = {1..#l.units, 1..#l.inputDim};
          l.outputDom = {1..#l.units, 1..#l.batchSize};
          fillRandom(l.W);
          var b: [{l.bias.domain.dim(1)}] real;
          fillRandom(b);
          [j in l.bias.domain.dim(2)] l.bias[..,j] = b;
        }
        l.predDom = {1..1, 1..#l.batchSize};
        if l.activation == nil {
          l.activation = new Activation("relu");
        }
        //writeln(" currentLayer ", l);
      }
      var topLayer = new Layer(units=yTrain.shape[1]);
    }
    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
        compile(xTrain, yTrain);
        var X = xTrain;

        var batchSize = xTrain.shape[1],
            inputDim = xTrain.shape[2],
            labelDim = yTrain.shape;

        var t: Timer;
        t.start();
        for e in 1..#epochs {
          for l in layers.domain {
            ref currentLayer = layers[l];
            if l == layers.domain.first {
              writeln(" currentLayer.W domain ", currentLayer.W.domain);
              writeln(" X.T domain ", X.T.domain);
              currentLayer.a = matPlus(currentLayer.bias, dot(currentLayer.W, X.T));
              continue;
            }
            writeln(" FP: On layer %i".format(l));
            ref lowerLayer = layers[l-1];
            writeln(" currentLayer.W domain ", currentLayer.W.domain);
            writeln("  lowerLayer.h domain ", lowerLayer.h.domain);
            var x = dot(currentLayer.W, lowerLayer.h);
          }
          for l in layers.domain by -1 {
            writeln(" BP: On layer %i".format(l));
          }
          writeln("Epoch (%i) error: ", layers[layers.domain.last].error);
        }
        t.stop();
        writeln(" elapsed time: ", t.elapsed());
        //return MLPfit(layers=this.layers, xTrain, yTrain, epochs, lr);
        return 0;
    }
  }

  class Dense {
    var units: int,
        batchSize: int,
        inputDim: int;
  }
  class Activation {
    var name: string;

    proc f(x: real) {
       if name == "relu" {
         return sigmoid(x);
       } else {
         return x;
       }
     }

     proc df(x:real) {
       if name == "relu" {
         return derivativesSigmoid(x);
       } else {
         return 1;
       }
     }

     proc sigmoid(x: real) {
       return (1/(1 + exp(-x)));
     }
     proc derivativesSigmoid(x) {
       return x * (1-x);
    }
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

    proc summarize() {
      writeln("Summary of layer %i".format(layerId));
      writeln("\tinputDim: %i\n\tunits: %i\n\tbatchSize: %i".format(inputDim,units, batchSize ));
      writeln("\tW size ", W.shape);
      writeln("\th size ", h.shape);
      writeln("\ta size ", a.shape);
      writeln("\tb size ", b.shape);
      writeln("\tbias size ", bias.shape);
      writeln("\tActivation ", activation.name);
    }

  }
  class Loss {

  }

}
