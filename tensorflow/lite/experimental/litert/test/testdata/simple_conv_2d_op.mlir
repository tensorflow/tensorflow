module {
func.func @main(%arg0: tensor<1x216x288x24xf32>, %arg1: tensor<24x3x3x24xf32>, %arg2: tensor<24xf32>) -> tensor<1x216x288x24xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x216x288x24xf32>, tensor<24x3x3x24xf32>, tensor<24xf32>) -> tensor<1x216x288x24xf32>
  return %0 : tensor<1x216x288x24xf32>
}
}