/*
 This is a pretty good example of the pseudocode http://www.cleveralgorithms.com/nature-inspired/neural/backpropagation.html
 */
module NN {
  use LinearAlgebra,
      Time,
      Random;

  class Sequential {
    var layerDom = {1..0},
        loss = new Loss(),
        layers: [layerDom] Layer;

    proc add(d: Dense) {
      writeln(" Adding a new dense layer");
      layers.push_back(new Layer());
      ref currentLayer = layers[layerDom.last];
      currentLayer.layerId = layerDom.last;
      if currentLayer.layerId == layerDom.first {
        currentLayer.units = d.units;
        currentLayer.inputDim = d.inputDim;
        currentLayer.batchSize = d.batchSize;
        currentLayer.weightDom = {1..#d.inputDim, 1..#d.units};
        currentLayer.outputDom = {1..#d.batchSize, 1..#d.units};
      }
      currentLayer.units = d.units;
    }

    proc add(d: Activation) {
      writeln(" Adding a new activation layer");
      ref currentLayer = layers[layerDom.last];
      currentLayer.activation = new Activation(name=d.name);
    }

    proc compile(xTrain:[], yTrain:[]) {
      // Add the output layer for calculating epoch error
      layers.push_back(new Layer());
      ref topLayer = layers[layerDom.last];
      topLayer.units = 1;
      topLayer.layerId = layerDom.last;

      for l in layers {
        if l.layerId == layerDom.first {
          fillRandom(l.W);
          var b: [{l.bias.domain.dim(1)}] real;
          fillRandom(b);
          [j in l.bias.domain.dim(2)] l.bias[..,j] = b;
        } else {
          ref lowerLayer = layers[l.layerId-1];
          l.inputDim = lowerLayer.units;
          l.batchSize = lowerLayer.batchSize;
          l.weightDom = {1..#l.inputDim, 1..#l.units};
          l.outputDom = {1..#l.batchSize, 1..#l.units};
          fillRandom(l.W);
          l.W = l.W / 25.0;
          var b: [{l.bias.domain.dim(1)}] real;
          fillRandom(b);
          [j in l.bias.domain.dim(2)] l.bias[..,j] = b;
        }
        if l.activation == nil {
          l.activation = new Activation("relu");
        }
      }
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
              currentLayer.a = matPlus(currentLayer.bias, dot(X, currentLayer.W));
              currentLayer.h = currentLayer.activation.f(currentLayer.a);
              continue;
            }
            //writeln("** FORWARDS : On layer %i".format(currentLayer.layerId));
            ref lowerLayer = layers[l-1];
            currentLayer.a = matPlus(currentLayer.bias, dot(lowerLayer.h, currentLayer.W));
            currentLayer.h = currentLayer.activation.f(currentLayer.a);
          }
          /*
            ***** BACKWARDS *****
           */
          for l in layers.domain by -1 {
            //writeln("** BACKWARDS at layer %i".format(l));
            ref currentLayer = layers[layerDom.last];
            ref lowerLayer = layers[layerDom.last-1];
            currentLayer.gradH = currentLayer.activation.df(currentLayer.h);
            // set the error
            if l == layerDom.last {
              currentLayer.error = loss.L(yTrain, currentLayer.h);
            } else {
              ref ul = layers[l+1];
              currentLayer.error = ul.h * currentLayer.gradH;
            }
            currentLayer.dH = currentLayer.error * currentLayer.gradH;
            currentLayer.W += dot(lowerLayer.h.T, currentLayer.dH) * lr;
            var ones: [currentLayer.dH.domain] real = 1.0;
            currentLayer.bias += dot(currentLayer.dH, ones.T) * lr;
          }
        }
        ref topLayer = layers[layerDom.last];
        t.stop();
        writeln("Completed %i epochs".format(epochs));
        writeln("  Predictions  : ", topLayer.h.T);
        writeln("  Final error  : ", topLayer.error.T);
        writeln("  max(error)   : ", max reduce abs(topLayer.error));
        writeln("  elapsed time : ", t.elapsed());
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
    W: [weightDom] real,
    a: [outputDom] real,
    h: [outputDom] real, // will be used in NEXT layer
    gradH: [outputDom] real, // the gradient of the output
    dH: [outputDom] real, // will be used in NEXT layer
    bias: [outputDom] real, // bias = [b,b,..]
    error: [outputDom] real;

    proc summarize() {
      writeln("Summary of layer %i".format(layerId));
      writeln("\tinputDim: %i\n\tunits: %i\n\tbatchSize: %i".format(inputDim,units, batchSize ));
      writeln("\tW size ", W.shape);
      writeln("\th size ", h.shape);
      writeln("\ta size ", a.shape);
      writeln("\tbias size ", bias.shape);
      writeln("\tActivation ", activation.name);
    }

  }
  class Loss {
    proc L(y:[], x:[]) {
      var yd: [x.domain] real;
      yd[..,1] = y;
      var e = matMinus(yd, x);
      return e;
    }
  }

}
