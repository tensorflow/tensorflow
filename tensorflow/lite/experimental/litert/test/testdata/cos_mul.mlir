module {
func.func @main(%arg0: tensor<8x100x32x2xf32>, %arg1: tensor<8x100x1x2xf32>) -> tensor<8x100x32x2xf32> {
  %0 = "tfl.cos"(%arg1) : (tensor<8x100x1x2xf32>) -> tensor<8x100x1x2xf32>
  %1 = tfl.mul(%arg0, %0) <{fused_activation_function = "NONE"}> : (tensor<8x100x32x2xf32>, tensor<8x100x1x2xf32>) -> tensor<8x100x32x2xf32>
  return %1 : tensor<8x100x32x2xf32>
}
}