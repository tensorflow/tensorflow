module {
func.func @main(%arg0: tensor<1x128x8x256xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<1x128x8x128xf32> {
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<1x128x8x256xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x128x8x128xf32>
  return %0 : tensor<1x128x8x128xf32>
}
}