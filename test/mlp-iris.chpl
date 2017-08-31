//use NNModels;
use NN;
writeln("How do you run tests in Chapel?");

writeln("Hola Mundo!");
var X = Matrix(
   [1.0,0.0,1.0,0.0],
   [1.0,0.0,1.0,1.0],
   [0.0,1.0,0.0,1.0]);
var y = Vector([1.0,1.0,0.0]);

const epochs:int = 5000,
      lr: real = 0.1;
var model = new Sequential();
model.add(new Dense(units=2, inputDim=4, batchSize=3));
model.add(new Dense(units=1));
//model.add(new Dense(units=6));
model.add(new Activation(name="relu"));

var o = model.fit(xTrain=X,yTrain=y, epochs=epochs, lr=lr);
writeln(o);
