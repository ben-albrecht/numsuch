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
    var y = Vector([1,1,0]);

    // Variable Initialization
    const inputLayerNeurons = X.shape[1],
          hiddenlayer_neurons = 3,
          output_neurons = 1;

    /* Create Ranges for the objects */
    const inputRange = 0..X.shape[1],
          hiddenRange = 0..3,
          outputRange = 0..#1;

    // Weight and bias matrices
    var wh: [{inputRange, hiddenRange}] real,
        bh: [{0..0, hiddenRange}] real,
        wout: [{hiddenRange, outputRange}] real,
        bout: [{0..0, outputRange}] real;

    fillRandom(wh);
    fillRandom(bh);
    fillRandom(wout);
    fillRandom(bout);

    for i in 1..#epoch {

      /* Forward propagation */
      var hidden_layer_input1 = dot(X,wh);
      writeln(hidden_layer_input1);
    }

  }

  proc sigmoid(x: real) {
    return (1/(1 + exp(-x)));
  }

  proc derivatives_sigmoid(x) {
    return x * (1-x);
  }
}
