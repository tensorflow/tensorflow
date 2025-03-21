module {
func.func @main(%input: tensor<1x1920x1080x3xf32>) -> tensor<1x480x540x3xf32> {
  %output = "tfl.max_pool_2d"(%input) {
    filter_height = 4 : i32,
    filter_width = 2 : i32,
    fused_activation_function = "NONE",
    padding = "VALID",
    stride_h = 4 : i32,
    stride_w = 2 : i32
  } : (tensor<1x1920x1080x3xf32>) -> tensor<1x480x540x3xf32>
  return %output : tensor<1x480x540x3xf32>
}
}