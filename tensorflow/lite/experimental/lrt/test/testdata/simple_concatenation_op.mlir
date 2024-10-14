module {
func.func @main(%arg0: tensor<128x4x1x256xf32>, %arg1: tensor<128x4x1x256xf32>) -> tensor<128x4x2x256xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1) <{axis = 2 : i32, fused_activation_function = "NONE"}> : (tensor<128x4x1x256xf32>, tensor<128x4x1x256xf32>) -> tensor<128x4x2x256xf32>
  return %0 : tensor<128x4x2x256xf32>
}
}