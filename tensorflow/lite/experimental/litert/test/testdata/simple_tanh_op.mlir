module {
func.func @main(%arg0: tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xf32> {
  %0 = "tfl.tanh"(%arg0) : (tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xf32>
  return %0 : tensor<1x128x8x128xf32>
}
}