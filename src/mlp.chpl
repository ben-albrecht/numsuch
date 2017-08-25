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
          hiddenRange = 0..#3,
          outputRange = 0..#1;

    /* Weight and bias matrices
       Note that inputRange counts the number of inputs, not the dimension
       of the input.  We'll have to clarify that.
     */
    var wh: [{inputRange, hiddenRange}] real,
        //bh: [{0..0, hiddenRange}] real,
        wout: [{hiddenRange, outputRange}] real,
        bout: [{0..0, outputRange}] real;

    var bh: [hiddenRange] real;

    fillRandom(wh);
    fillRandom(bh);
    fillRandom(wout);
    fillRandom(bout);

    var bhm: [hiddenRange, hiddenRange] real;
    for i in hiddenRange{
      bhm[i,..] = bh;
    }



    for i in 1..#epoch {

      /* Forward propagation */
      //writeln("X shape: ", X.shape);
      //writeln("wh shape: ", wh.shape);
      //writeln("bh shape: ", bh.shape);
      //writeln("bhm shape: ", bhm.shape);
      //writeln("bhm:\n ", bhm);
      var hidden_layer_input1 = dot(X,wh);
      //writeln(hidden_layer_input1.shape);
      //writeln(bh.shape);
      //writeln(bhm.shape);
      /* Here, Python pulls a bullshit move
         It adds b to each ROW of dot(X,wh), which is bullshit any way you slice it.
       */
      //var hidden_layer_input = matPlus(hidden_layer_input1, bs);
      var hidden_layer_input = matPlus(hidden_layer_input1, bhm);
    }

  }

  proc sigmoid(x: real) {
    return (1/(1 + exp(-x)));
  }

  proc derivatives_sigmoid(x) {
    return x * (1-x);
  }
}
