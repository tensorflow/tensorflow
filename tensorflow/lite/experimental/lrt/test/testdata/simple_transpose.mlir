module {
func.func @main(%arg0: tensor<128x4x2x128xf32>, %arg1: tensor<4xi32>) -> tensor<2x128x4x128xf32> {
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<128x4x2x128xf32>, tensor<4xi32>) -> tensor<2x128x4x128xf32>
  return %0 : tensor<2x128x4x128xf32>
}
}