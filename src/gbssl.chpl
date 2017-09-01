/*
  Package of Graph-Based Semi-Supervised Learning techniques based partially
  on the work by `Subramanya and Talukdar <http://graph-ssl.wdfiles.com/local--files/blog%3A_start/graph_ssl_acl12_tutorial_slides_final.pdf>`_

  Also, see notes in the /tex folder under main.

 */
module GBSSL {
  use LinearAlgebra;

  class ModifiedAdsorptionModel {
    var vdom: domain(1),
        gdom: domain(2),
        edom: sparse subdomain(gdom),
        v: [vdom] real,
        e: [edom] real,
        directed: bool,
        A: [gdom] real,
        pcont: [vdom] real,
        pabdn: [vdom] real,
        pinj : [vdom] real,
        compiled: bool = false,
        beta: real = 1.1;

    proc fit(data: [], labels: []) {
      if !compiled {
        compile();
      }
      if data.shape[1] != data.shape[2] {
      } else {
        vdom = {1..#data.shape[1]};
        gdom = {1..#data.shape[1], 1..#data.shape[1]};
        var xd = data.domain;
        ref Xd = A.reindex(xd);
        for ij in data.domain {
          if data[ij] > 0 {
            edom += ij+(1,1);
            Xd[ij] = data[ij];
          }
        }
      }
    }

    /*
      We need three probabilities as each vertex for the MAD algo.
     */

    proc compile() {
      calculateProbs();
      compiled = true;
    }
    proc calculateProbs() {
      for v in vdom {
        var ps = cellProbabilities(v);
        pcont[v] = ps[1];
        pinj[v] = ps[2];
        pabdn[v] = ps[3];
      }
    }

    /*
     Try to take advantage of function promotion by doing this one element at a time
     Will need some expert advice.
     */
     proc cellProbabilities(i:int) {
       // Need to remove the diagonal from the expressions m and l below
       var m: real = + reduce A[i,..];
       m = m - A[i,i];
       var l = + reduce xlogx(A[i,..]);
       l = l - xlogx(A[i,i]);
       var h = max(log(m) - l / m,0);
       var c = log(beta) / (log(beta + h));
       var d = (1- c)* sqrt(h);
       var z = max(c+d, 1);
       //writeln(" cell probabilities (%n, %n, %n)".format(c/z, d/z, 1-(c+d)/z));
       return (c/z, d/z, 1-c-d);
     }
  }

   proc xlogx(x: real) {
     if x > 0.0005 {
       return x * log(x);
     } else {
       return 0;
     }
   }
}
