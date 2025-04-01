module {
func.func @main(%arg0: tensor<1x128x8x128xi1>, %arg1: tensor<1x128x8x128xf32>, %arg2: tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xf32> {
  %0 = "tfl.select"(%arg0, %arg1, %arg2) : (tensor<1x128x8x128xi1>, tensor<1x128x8x128xf32>, tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xf32>
  return %0 : tensor<1x128x8x128xf32>
}
}