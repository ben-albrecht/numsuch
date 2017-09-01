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
        A: [gdom] real,
        pContinue: [edom] real;

    proc add(x: []) {
      if x.shape[1] != x.shape[2] {
      } else {
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
      }
      pContinue();
    }

    proc pContinue() {
      for i in vdom {
        writeln(i);
        var denom = + reduce A[i,..] - A[i,i];
        for j in vdom {
            pContinue[i,j] = A[i,j] / denom;
        }
        writeln(denom);
        continue;
      }
    }

    proc calculatePContinue(ij) {
      var denom  = + reduce A[ij[1],..] - A[ij[1], ij[1]];
      var w = A[ij[1], ij[1]] / denom;
      return w;
    }


  }
}
