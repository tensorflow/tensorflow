module {
func.func @main(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<4xi32>) -> tensor<128x4x1x256xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<1x128x4x256xf32>, tensor<4xi32>) -> tensor<128x4x1x256xf32>
  return %0 : tensor<128x4x1x256xf32>
}
}