module MLP {
  // https://gist.github.com/ben-albrecht/e2765ddd49007744a09dd016535038f9

  use LinearAlgebra;
  use Random;
  //config const epoch = 5000;
  config const epoch = 5,
               lr = 0.1;

  /*
   Attempts to implement the Multi-Layer Perceptron described here
   https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
   I'm going to do it very literally the first time
   */
  proc main() {
    writeln("Hola Mundo!");
    var X = Matrix(
       [1.0,0.0,1.0,0.0],
       [1.0,0.0,1.0,1.0],
       [0.0,1.0,0.0,1.0]);
    var y = Vector([1.0,1.0,0.0]);

    // Variable Initialization
    const inputLayerNeurons = X.shape[1],
          hiddenlayer_neurons = 3,
          output_neurons = 1;

    /* Create Ranges for the objects */
    const nobsRange = 0..#X.shape[1], // number of observations
          ftrDimRange = 0..#X.shape[2], // Width of observations
          hiddenRange = 0..#3,
          outputRange = 0..#1;

    var ones: [hiddenRange] real = 1.0;
    var wh: [{ftrDimRange, hiddenRange}] real,
        wout: [{hiddenRange, outputRange}] real;

    var bh: [hiddenRange] real;
    var bout: [outputRange] real;

    fillRandom(wh);
    fillRandom(bh);
    fillRandom(wout);
    fillRandom(bout);

    var bhm: [hiddenRange, hiddenRange] real;
    [i in hiddenRange] bhm[i,..] = bh;

    /* Forward propagation variables, set domains */
    var hiddenLayerInput1: [nobsRange, hiddenRange] real;
    var hiddenLayerInput: [nobsRange, hiddenRange] real;
    var hiddenLayerActivations: [nobsRange, hiddenRange] real;
    var outputLayerInput1: [nobsRange, outputRange] real;
    var outputLayerInput: [nobsRange, outputRange] real;
    var output: [nobsRange, outputRange] real;
    var E: [nobsRange, outputRange] real;
    var slopeOutputLayer: [nobsRange, outputRange] real;
    var slopeHiddenLayer: [nobsRange, hiddenRange] real;
    var dOutput: [nobsRange, outputRange] real;
    var errorAtHiddenLayer: [nobsRange, hiddenRange] real;
    var dHiddenLayer: [nobsRange, hiddenRange] real;
    for i in 1..#epoch {
      /* Forward propagation */
      hiddenLayerInput1 = dot(X,wh);
      /* Here, Python pulls a bullshit move
         It adds b to each ROW of dot(X,wh), which is bullshit any way you slice it.
       */
      hiddenLayerInput = matPlus(hiddenLayerInput1, bhm);
      hiddenLayerActivations = sigmoid(hiddenLayerInput);
      outputLayerInput1 = dot(hiddenLayerActivations, wout);
      outputLayerInput = matPlus(outputLayerInput1, wout);
      output = sigmoid(outputLayerInput);

      /* Backward propagation */
      E = y - output;
      slopeOutputLayer = derivatives_sigmoid(output);
      slopeHiddenLayer = derivatives_sigmoid(hiddenLayerActivations);
      /*
        This freaks me out, this does element-wise multiplication.
       */
      dOutput = E * slopeOutputLayer;
      errorAtHiddenLayer = dot(dOutput, wout.T);
      dHiddenLayer = dot(errorAtHiddenLayer,slopeHiddenLayer);

      /* Updates */
      wout += dot(hiddenLayerActivations.T, dOutput) * lr;
      bout += dot(dOutput.T, ones) * lr;
      wh += dot(X.T, dHiddenLayer) * lr;
      bh += dot(dHiddenLayer, ones.T) * lr;
      [i in hiddenRange] bhm[i,..] = bh;
    }
  }

  proc sigmoid(x: real) {
    return (1/(1 + exp(-x)));
  }

  proc derivatives_sigmoid(x) {
    return x * (1-x);
  }
}
