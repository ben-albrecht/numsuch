Here we put scripts in other language, often Python, to help build out the algs.

# MLP Routine

I used this [simple example](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/) to begin with.

| Tutorial Name | Symbol | Formulation |
----------------------------------------
| X | h0 | given |
| y | y | given |
| wh | W1 | random |
| wout | W2 | random |
| bh | b1 | random |
| bout | b2 | random |
| hidden layer input 1 |  | dot(X, W1) |
| hidden layer input | a1 | dot(X, W1) + b1 |
| hidden layer activations | h1 | sigmoid(W1 h0 + b1) = sigmoid(a1) |
| output layer input 1 |  | dot(h1, W2) |
| output layer input | a2 | dot(h1, W2) + b2 |
| output | h2 | sigmoid(a2) |
| E | E2 | y - h2 |
| slope output layer | gradH2 | dsigma(h2) |
| d output | dH2 | E2 * gradH2 |
| slope hidden layer | gradH1 | dsigma(h1) |
| error at hidden layer | E1 | dH2 W2^t |
| d hidden layer | dH1 | E1 * gradH1 |
| wout += | W2 | += dot(h1^t, dH2) * lr |
| bout += | b2 | += dot(dH2,1) * lr |
| wh += | W1 | += dot(X^t, dH1) |
| bh += | b1 | += dot(dH1, 1) * lr |

## Forward

```
h0 = X
i = 1..2:
  ai = bi + Wi h(i-1)
  hi = sigmoid(ai)
yhat = h2
```

## Backwards

```
E2 = y - yHat
i = 2..1:
  gradHi = dsigma(hi)
  Ei     = y-h2 if i == 2
           h(i+1)^t * gradHi o/w
  dHi    = Ei * gradHi
  Wi    += dot(h(i-1)^t, dHi) * lr
  bi    += dot(dHi, 1) * lr  
```

Or

```
E2 = y - yHat
i = 2..1:
  gradHi = dsigma(hi)
  Ei     = y-h2 if i == 2
           h(i+1)^t * gradHi o/w
  dHi    = Ei * gradHi
  Wi    += dot(h(i-1), dHi^t) * lr
  bi    += dot(dHi, 1) * lr  
```
