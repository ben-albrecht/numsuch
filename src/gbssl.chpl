/*
  Package of Graph-Based Semi-Supervised Learning techniques based partially
  on the work by `Subramanya and Talukdar <http://graph-ssl.wdfiles.com/local--files/blog%3A_start/graph_ssl_acl12_tutorial_slides_final.pdf>`_

 */
module GBSSL {
  use LinearAlgebra;

  class AdjacencyMatrix {
    var vdom: domain(1),
        gdom: domain(2),
        edom: sparse subdomain(gdom),
        v: [vdom] int,
        e: [edom] real,
        directed: bool,
        A: [gdom] real;

    proc add(x: []) {
      if x.shape[1] != x.shape[2] {
        writeln("Error!  X must be square!");
      } else {
        writeln("  x.domain ", x.domain);
        vdom = {1..#x.shape[1]};
        gdom = {1..#x.shape[1], 1..#x.shape[1]};
        var xd = x.domain;
        ref Xd = A.reindex(xd);
        for ij in x.domain {
          if x[ij] > 0 {
            edom += ij+(1,1);
            Xd[ij] = x[ij];
          }
        }
        writeln(A);
      }
    }

    proc pContinue() {
      for ij in A.domain {
        continue;
      }

    }
  }
}
