module NNModels {
  use MLP;

  /*
    Used to hold Deep Learning workflows with their multiple layers.
    Add layers to the model then run model.fit(<params>)
   */
  class Sequential {
    var epochs: int,
        lr: real;

    var layers = {0..0};
    var denseLayers: [layers] Dense;
    proc add(d: Dense) {
      denseLayers[layers.size-1] = d;
      // @TODO don't add a layer now
      layers = {0..#layers.size+1};
    }

    proc fit(xTrain:[], yTrain:[], epochs:int, lr: real) {
      return mlpFit(X=xTrain, y=yTrain, epochs=epochs, lr=lr);
    }
  }

  class Dense {
    var units: int,
        inputDim: int;
  }
}
