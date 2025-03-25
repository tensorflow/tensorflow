module {
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<32x2x2x32xf32>, %arg2: tensor<1x128x128x32xf32>, %arg3: tensor<32xf32>) -> tensor<1x256x256x32xf32> {
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %arg3) <{dilation_h_factor = 4 : i32, dilation_w_factor = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<4xi32>, tensor<32x2x2x32xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x256x256x32xf32>
  return %0 : tensor<1x256x256x32xf32>
    }
}
 