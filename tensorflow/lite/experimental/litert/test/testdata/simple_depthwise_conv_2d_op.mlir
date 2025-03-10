module {
func.func @main(%arg0: tensor<1x40x40x192xf32>, %arg1: tensor<1x3x3x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x32x32x192xf32> {
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) <{depth_multiplier = 1 : i32, dilation_h_factor = 4 : i32, dilation_w_factor = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x40x40x192xf32>, tensor<1x3x3x192xf32>, tensor<192xf32>) -> tensor<1x32x32x192xf32>
  return %0 : tensor<1x32x32x192xf32>
}
}