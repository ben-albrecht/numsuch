use NNModels;
writeln("How do you run tests in Chapel?");

writeln("Hola Mundo!");
var X = Matrix(
   [1.0,0.0,1.0,0.0],
   [1.0,0.0,1.0,1.0],
   [0.0,1.0,0.0,1.0]);
var y = Vector([1.0,1.0,0.0]);

const epochs:int = 4,
      lr: real = 0.1;
var model = new Sequential(epochs=epochs, lr=lr);
model.add(new Dense(units=2, inputDim=4, batchSize=3));
model.add(new Dense(units=4));
model.add(new Dense(units=9));
model.add(new Activation(name="relu"));

var o = model.fit(X,y, epochs, lr);
writeln(o);
