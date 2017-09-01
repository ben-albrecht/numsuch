/*
 Designed to test the Modified Adsoprtion routine
 */
use GBSSL;

var X = Matrix(
  [2.0, 0.2,  .0, 1.5],
  [0.2, 1.0,  .0,  .0],
  [ .0, 1.5, 2.0, 1.2],
  [1.5,  .0, 1.2,  1.0]
);

var y = Vector([1.0,1.0,0.0,1.0]);

var A = new AdjacencyMatrix();
A.add(X);
