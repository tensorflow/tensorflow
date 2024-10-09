module {
func.func @main(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<1xi32>) -> tensor<1x128x1xf32> {
  %0 = "tfl.sum"(%arg0, %arg1) <{keep_dims = true}> : (tensor<1x128x2304xf32>, tensor<1xi32>) -> tensor<1x128x1xf32>
  return %0 : tensor<1x128x1xf32>
}
}
