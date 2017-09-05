module Core {

  /*
   Class to hold labels for data.  Has names in names
   */
  class LabelMatrix {
    var ldom: domain(1) = {1..0},
        dataDom: domain(2),
        data: [dataDom] real,
        names: [ldom] string;
    /*
    Loads a label file into a Matrix.  Labels should be binary indicators
    <record id: string> <category 1 indicator> ... <category L indicator>
     */
    proc readFromFile(fn: string, addDummy: bool = false) {
      var lFile = open(fn, iomode.r).reader(),
          x: [1..0] real,
          nFields: int,
          line: string,
          //ldom: domain(1),
          xline: [ldom] real,
          nRows: int = 1,
          firstLine: bool = true;
      for line in lFile.lines() {
         var fields = line.split("\t");
         if firstLine {
           nFields = fields.size;
           if addDummy {
             // Add a dummy column
             ldom = {1..nFields};
           } else {
             ldom = {1..nFields-1};
           }
           dataDom = {1..0, 1..nFields};
           firstLine = false;
         } else {
           if fields.size != nFields {
             halt("Unequal number of fields in label file");
           }
         }
         //names.push_back(fields[1]);
         writeln(ldom);
         ldom = {1..ldom.last+1};
         //names.push_back(fields[1]);
         names[ldom.last] = fields[1];
         dataDom = { 1..#nRows, ldom.dim(1)};
         var xline: [ldom] real;
         for v in 2..fields.size {
           xline = fields[v]:real;
         }
         data[dataDom.last(1), ..] = xline;
         nRows += 1;
      }
    }
  }

  /*
   Does the pairwise cosine distance between rows of X and rows of Y.
   Dimensions must be conformable
   */
  proc cosineDistance(X:[], Y:[], denseOutput=true) {
    if X.shape[2] != Y.shape[2] {
      halt(" dimension mismatch: X = (%n,%n)  Y = (%n,%n)".format(X.shape[1], X.shape[2], Y.shape[1], Y.shape[2]));
    }
    // Make magic happen

  }
}
