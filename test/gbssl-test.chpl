/*
 Designed to test the Modified Adsoprtion routine
 */
use GBSSL,
    MatrixMarket;


var X = Matrix(
  [3.0, 0.2, 0.0, 0.7, 0.1],
  [0.2, 2.0, 0.3, 0.0, 0.0],
  [0.0, 0.3, 3.0, 0.9, 0.6],
  [0.7, 0.0, 0.9, 2.0, 0.0],
  [0.1, 0.0, 0.6, 0.0, 2.0]
);

var y = Matrix(
  [1,0,0],
  [0,1,0],
  [0,0,1],
  [1,0,1],
  [0,0,0]
  );

var labelFile = "data/webkb_labels.txt";
var vectorFile = "data/webkb_vectors.mtx";
var L: LabelMatrix = labelsToMatrix(labelFile,addDummy=true);
var W = mmread(real, vectorFile);
var V = vectorsToAdjacency(W);

/*
var model = new ModifiedAdsorptionModel();
model.fit(data=X, labels=y);
writeln(model.A);
 */
