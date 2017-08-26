module NNModels {
  //use MLP;
  use LinearAlgebra;
  use Random;

  /*
    Used to hold Deep Learning workflows with their multiple layers.
    Add layers to the model then run model.fit(<params>)

    It is important to add Dense layer, then activation.
   */
  class Sequential {
    var epochs: int,
        lr: real;

    var layerDom: domain(1) = {0..0}; // Is now 1 long
    var layers: [layerDom] Layer;
    var denseLayers: [layerDom] Dense;
    var activationLayers: [layerDom] Activation;
    var inputDims: [layerDom] int;
    var units: [layerDom] int;
    //var Ws: [layerDom] Matrix;
    proc add(d: Dense) {
      // Is it the first?
      if denseLayers[0] == nil {
        writeln("  Very first dense layer");
        var l = new Layer(dataLayer=d, inputDim=d.inputDim, units=d.units);
        denseLayers[0] = d;
        layers[0] = l;
      } else {
        writeln("  Adding dense layer %i".format(layerDom.size));
        layerDom = {0..#layerDom.size+1};
        denseLayers[layerDom.size-1] = d;
        var l = new Layer(dataLayer=d, inputDim=d.inputDim, units=d.units);
        layers[layerDom.size-1] = l;
      }
    }
    /*
     Must be added sequentially
     */
    proc add(a: Activation) {
      writeln("  Adding activation to layer %i".format(layers.size-1));
      //layers[layerDom.size-1].activation = a;
      activationLayers[layerDom.size-1] = a;
    }

    /*
     */
    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
      //return MLP.fit(X=xTrain, y=yTrain, epochs=epochs, lr=lr);
      writeln("Initialization of weights");
      for l in 0..#layers.size {
        var wTmp = layers[l].W;
        fillRandom(layers[l].W);
        writeln(layers[l].W);

        //Ws[l] = wTmp;
      }
      for e in 1..#epochs {
          writeln("epoch %i".format(e));
          for l in 0..#layers.size {
            writeln("...in layer %i".format(l));

          }
      }
      return 0;
    }
  }

  /*
  inputDim: dimension of the input
  units: dimension of the output
   */
  class Layer {
    var dataLayer: Dense,
        activation: Activation,
        inputDim: int,
        units: int,
        W: [{0..#inputDim, 0..#units}] real,
        bias: [{0..0}] real;
  }

  /*
   Holds a layer of data.
   */
  class Dense {
    var units: int,  // dimension of the output space, to be fed to the next layer
        inputDim: int,  // Size of the vectors being considered
        batch_size: int;  // How many vectors are being sent it
  }

  class Activation {
    var name: string;

    /* @TODO abstract function and derivative to f() and df() */
    proc sigmoid(x: real) {
      return (1/(1 + exp(-x)));
    }

    proc derivativesSigmoid(x) {
      return x * (1-x);
    }
  }
}
