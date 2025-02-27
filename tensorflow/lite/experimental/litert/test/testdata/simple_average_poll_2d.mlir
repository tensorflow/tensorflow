module {
func.func @main(%arg0: tensor<1x1728x2304x3xf32>) -> tensor<1x432x576x3xf32> {
  %0 = "tfl.average_pool_2d"(%arg0) <{filter_height = 4 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 4 : i32}> : (tensor<1x1728x2304x3xf32>) -> tensor<1x432x576x3xf32>
  return %0 : tensor<1x432x576x3xf32>
}
}