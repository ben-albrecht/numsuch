module MLP {
  use LinearAlgebra;
  use Random;
  config const epoch = 5000;
  config const lr = 0.1;

  /*
   Attempts to implement the Multi-Layer Perceptron described here
   https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
   I'm going to do it very literally the first time
   */
  proc main() {
    writeln("Hola Mundo!");
    var dom = {1..3,1..4};
    var X: [dom] real;
    X = Matrix([1,0,1,0],
       [1,0,1,1],
       [0,1,0,1]);
    var y = Vector([1,1,0]);

    // Variable Initialization
    var inputlayer_neurons = X.domain.dim(1).size;
    var hiddenlayer_neurons = 3;
    var output_neurons = 1;

    // Weight and bias matrices
    var wh: [{1..#inputlayer_neurons,1..#hiddenlayer_neurons}] real;
    var bh: [{1..1,1..#hiddenlayer_neurons}] real;
    var wout: [{1..#hiddenlayer_neurons, 1..#output_neurons}] real;
    var bout: [{1..1, 1..#output_neurons}] real;
    fillRandom(wh);
    fillRandom(bh);
    fillRandom(wout);
    fillRandom(bout);

    

  }

  proc sigmoid(x: real) {
    return (1/(1 + exp(-x)));
  }

  proc derivatives_sigmoid(x) {
    return x * (1-x);
  }
}
