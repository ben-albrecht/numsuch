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
    var needSomeLayers = true;
    proc add(d: Dense) {
      // Is it the first?
      var curidx = layerDom.size-1;
      // layerDom has a default size of 1, even if empty
      if curidx == 0 && needSomeLayers {
        writeln("  Very first dense layer");
        var l = new Layer(dataLayer=d, inputDim=d.inputDim, units=d.units);
        layers[0] = l;
        needSomeLayers = false;
      } else {
        writeln("  Adding dense layer %i".format(layerDom.size));
        layerDom = {0..#layerDom.size+1};
        if d.units < 1 {
          writeln(" Error, need units > 0");
        }
        var l = new Layer(dataLayer=d,inputDim=layers[curidx].units,units=d.units);
        layers[layerDom.size-1] = l;
      }
    }
    /*
     Must be added sequentially
     */
    proc add(a: Activation) {
      writeln("  Adding activation to layer %i".format(layers.size-1));
      layers[layerDom.size-1].activation = a;
    }

    /*
     */
    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
      writeln("Initialization of weights");
      for l in 0..#layers.size {
        writeln("Prepping layer %i".format(l));
        //writeln(layers[l].W.domain);
        fillRandom(layers[l].W);
        fillRandom(layers[l].bias);
        /*
        writeln("W:");
        writeln(layers[l].W);
        writeln("\nb:");
        writeln(layers[l].bias);
        */
      }
      /*
      for e in 1..#epochs {
          writeln("epoch %i".format(e));
          for l in 0..#layers.size {
            writeln("...in layer %i".format(l));

          }
      } */
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
        bias: [{0..#units}] real;
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
