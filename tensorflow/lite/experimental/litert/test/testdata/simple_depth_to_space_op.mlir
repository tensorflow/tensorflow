module {
func.func @main(%arg0: tensor<1x216x288x12xf32>) -> tensor<1x432x576x3xf32> {
  %0 = "tfl.depth_to_space"(%arg0) <{block_size = 2 : i32}> : (tensor<1x216x288x12xf32>) -> tensor<1x432x576x3xf32>
  return %0 : tensor<1x432x576x3xf32>
}
}