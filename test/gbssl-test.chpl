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

var y = Matrix(
  [1,0,0],
  [0,1,0],
  [0,0,1],
  [1,0,1]
  );

var model = new ModifiedAdsorptionModel();
model.fit(data=X, labels=y);
writeln(model.A);
