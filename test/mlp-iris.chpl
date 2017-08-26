use MLP;
writeln("How do you run tests in Chapel?");

writeln("Hola Mundo!");
var X = Matrix(
   [1.0,0.0,1.0,0.0],
   [1.0,0.0,1.0,1.0],
   [0.0,1.0,0.0,1.0]);
var y = Vector([1.0,1.0,0.0]);

var o = fit(X,y);
writeln(o);
