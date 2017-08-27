/*
 Is this a package level comment?

 Algorithm 6.3 (Feed Forward) from Bengio, et al.
 Require: Network depth l
 Require: W for 1...l
 Require: b for 1...l
 Require: x, input data
 Require: y, target output (e.g. labels)
 h_0 = x
 for k = 1..l do
  a_k = b_k + W_k h_{k-1}  // Notice, this is equiv to adding b to each ROW
  h_k = f(a_k) // f = activation function
 end for
 y = h_l
 J = L(y,y) + lambda * Omega(theta)  // described later

 */
module NNModels {
  //use MLP;
  use LinearAlgebra;
  use Random;

  class Sequential {
    var epochs: int,
        lr: real;

    var layerDom: domain(1) = {1..0}; // Is now 1 long
    var layers: [layerDom] Layer;
    var needSomeLayers = true;
    proc add(d: Dense) {
      // Is it the first?
      var curidx = layerDom.size;
      writeln("curidx ", curidx);
      // layerDom has a default size of 1, even if empty
      // So curidx will be 0 twice

      if curidx == 0 {
        layerDom = {1..#layerDom.size+1};
        writeln("  Very first dense layer %i".format(layerDom.size));
        var l = new Layer(dataLayer=d
          , inputDim=d.inputDim
          , units=d.units
          , activation = new Activation(name="linear"));
        layers[layerDom.size] = l;
        //writeln(layers[layerDom.size]);
      } else {
        layerDom = {1..#layerDom.size+1};
        writeln(" Adding Dense layer %i".format(layerDom.size));
        var l = new Layer(dataLayer=d
          ,inputDim=layers[layerDom.size-1].units
          ,units=d.units
          ,activation=new Activation(name="linear"));
        layers[layerDom.size] = l;
        //writeln(layers[layerDom.size]);
      }
    }
    /*
     Must be added sequentially
     */
    proc add(a: Activation) {
      writeln("  Adding activation to layer %i".format(layerDom.size));
      //writeln(layers[layerDom.size-1]);
      //writeln(layers[layerDom.size]);
      layers[layerDom.size].activation = a;
    }

    /*
     */
    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
      writeln("Initialization of weights");
      writeln(xTrain.domain);
      writeln(layers[1].W.domain);
      writeln(xTrain);
      writeln(layers[1].W);
      //layers[1].W = xTrain;
      //var wt = dot(xTrain, layers[1].W);
      for l in layerDom {
        if l == 1 {
          continue;
        }
        writeln("Prepping layer %i".format(l));
        //writeln(layers[l].W.domain);
        fillRandom(layers[l].W);
        fillRandom(layers[l].bias);
        writeln("W:");
        writeln(layers[l].W);
        //writeln("\nb:");
        //writeln(layers[l].bias);
      }
      for e in 1..#epochs {
          writeln("epoch %i".format(e));
          writeln("Going FORWARD");
          for l in layerDom {
            if l == 1 {
              continue;
            }
            writeln("%i: I need to \n\t1. multiply W_t-1, W_t\n\t2. Add the bias to each row\n\t3. Apply the activation function".format(l));
            writeln("W_last domain ", layers[l-1].W.domain);
            writeln("W_now domain ", layers[l].W.domain);
            var wl = dot(layers[l-1].W, layers[l].W);

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
        bias: [{0..#units}] real;
  }

  /*
   Holds a layer of data.
   */
  class Dense {
    var units: int,  // dimension of the output space, to be fed to the next layer
        inputDim: int,  // Size of the vectors being considered
        batchSize: int;  // How many vectors are being sent it
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
