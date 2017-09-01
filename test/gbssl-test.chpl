/*
 Designed to test the Modified Adsoprtion routine
 */
use GBSSL;

var X = Matrix(
  [2.0, 0.2,  .0,  .7],
  [0.2, 1.0, 0.3,  .0],
  [ .0, 0.3, 2.0,  .9],
  [0.7,  .0,  .9, 2.0]
);

var y = Vector([1.0,1.0,0.0,1.0]);

var A = new AdjacencyMatrix();
A.add(X);
A.calculateProbs();
writeln(A.A);
