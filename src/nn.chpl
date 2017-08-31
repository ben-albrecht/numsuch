/*
 This is a pretty good example of the pseudocode http://www.cleveralgorithms.com/nature-inspired/neural/backpropagation.html

 ** BACK PROP **
 Initial Alg dervided from several sources:
 y    : labels for target
 E    : Error at current layer
 E_   : Error at layer below
 h+   : output of layer above
 h    : output of current layer
 h_   : ouput of layer below
 gradH: gradient of current layer
 dH   : derivative of current output
 W    : current weight Matrix
 W_   : weight matrix for layer below
 b    : current bias Matrix
 lr   : learning rate
 *    : element-wise multiplication
 1    : appropriate sized one vector

 for l in top layer to bottom layer, observes 3 layers at a time
    gradH = derivative of activation on h
    E     = y - h  if top layer;
            h+ W
    dH    = E * gradH
    W    += h_ dH * lr
    b    += dH 1 * lr

 for l in top to bottom, observes two layers at a time
     gradH = derivative of activation on h
     E     = y - h  if top layer;
             h+ W
     dH    = E * gradH
     W    += h_ dH * lr
     b    += dH 1 * lr

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
      //var l = new Layer(units=d.units, inputDim=d.inputDim, layerId=layerDom.last);
      layers.push_back(new Layer());
      ref currentLayer = layers[layerDom.last];
      currentLayer.layerId = layerDom.last;
      if currentLayer.layerId == layerDom.first {
        currentLayer.units = d.units;
        currentLayer.inputDim = d.inputDim;
        currentLayer.batchSize = d.batchSize;
        //currentLayer.weightDom = {1..#d.units, 1..#d.inputDim};
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

    //proc add(d: Activation) { }

    proc compile(xTrain:[], yTrain:[]) {
      // Add the output layer for calculating epoch error
      layers.push_back(new Layer());
      ref topLayer = layers[layerDom.last];
      topLayer.units = 1;
      topLayer.layerId = layerDom.last;

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
          l.weightDom = {1..#l.inputDim, 1..#l.units};
          l.outputDom = {1..#l.batchSize, 1..#l.units};
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
              //writeln(" X domain ", X.domain);
              //writeln(" currentLayer.W domain ", currentLayer.W.domain);
              //writeln(currentLayer.a.domain);
              //writeln(currentLayer.bias.domain);
              currentLayer.a = matPlus(currentLayer.bias, dot(X, currentLayer.W));
              currentLayer.h = currentLayer.activation.f(currentLayer.a);
              continue;
            }
            //writeln("** FORWARDS : On layer %i".format(currentLayer.layerId));
            ref lowerLayer = layers[l-1];
            //writeln("  lowerLayer.h domain ", lowerLayer.h.domain);
            //writeln("  currentLayer.W domain ", currentLayer.W.domain);
            currentLayer.a = matPlus(currentLayer.bias, dot(lowerLayer.h, currentLayer.W));
            currentLayer.h = currentLayer.activation.f(currentLayer.a);
          }
          /*
          writeln(" yTrain domain ", yTrain.domain);
          writeln(" currentLayer.h.domain ", currentLayer.h.domain);
          writeln(" currentLayer.error.domain ", currentLayer.error.domain);
           */
          /* At the top, calculate the error and set some values before descending */

          /*
            ***** BACKWARDS *****
           */
          for l in layers.domain by -1 {
            //writeln("** BACKWARDS at layer %i".format(l));
            ref currentLayer = layers[layerDom.last];
            ref lowerLayer = layers[layerDom.last-1];
            currentLayer.gradH = currentLayer.activation.df(currentLayer.a);
            // set the error
            if l == layerDom.last {
              writeln("Epoch (%i) error: ".format(e), currentLayer.error.T);
              //writeln("  currentLayer.gradH ", currentLayer.gradH);
              //writeln("  yTrain.domain ", yTrain.domain);
              //writeln("  currentLayer.h.domain ", currentLayer.h.domain);
              //writeln("  currentLayer.error.domain ", currentLayer.error.domain);
              currentLayer.error = loss.L(yTrain, currentLayer.h);
            } else {
              ref ul = layers[l+1];
              //writeln("   currentLayer.error.domain ", currentLayer.error.domain);
              //writeln("   uh.h.T.domain ", ul.h.T.domain);
              //writeln("   currentLayer.gradH.domain ", currentLayer.gradH.domain);
              currentLayer.error = ul.h * currentLayer.gradH;
            }
            currentLayer.dH = currentLayer.error * currentLayer.gradH;
            //writeln(" currentLayer.W.domain ", currentLayer.W.domain);
            //writeln(" currentLayer.dH.domain ", currentLayer.dH.domain);
            //writeln(" lowerLayer.h.T.domain ", lowerLayer.h.T.domain);
            //currentLayer.W += dot(currentLayer.dH, lowerLayer.h.T) * lr;
            currentLayer.W += dot(lowerLayer.h.T, currentLayer.dH) * lr;
            var ones: [currentLayer.dH.domain] real = 1.0;
            var b = dot(currentLayer.dH, ones.T) * lr;
            //writeln("   b.domain ", b.domain);
            //writeln("   currentLayer.b.domain ", currentLayer.b.domain);
            //writeln("   currentLayer.bias.domain ", currentLayer.bias.domain);
            //writeln("   currentLayer.dH.domain ", currentLayer.dH.domain);
            //currentLayer.b += dot(currentLayer.dH, ones.T) * lr;
          }
        }
        t.stop();
        writeln(" elapsed time: ", t.elapsed());
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
    gradH: [outputDom] real, // the gradient of the output
    dH: [outputDom] real, // will be used in NEXT layer
    b: [predDom] real, // The single column of bias
    bias: [outputDom] real, // bias = [b,b,..]
    error: [outputDom] real,
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
    proc L(y:[], x:[]) {
      //var yd: [{1..1,1..#y.shape[1]}] real;
      var yd: [x.domain] real;
      yd[..,1] = y;
      var e = matMinus(yd, x);
      return e;
    }
  }

}
