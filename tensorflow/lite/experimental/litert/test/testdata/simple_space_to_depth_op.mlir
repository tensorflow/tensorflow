module {
func.func @main(%arg0: tensor<1x432x576x6xf32>) -> tensor<1x216x288x24xf32> {
  %0 = "tfl.space_to_depth"(%arg0) <{block_size = 2 : i32}> : (tensor<1x432x576x6xf32>) -> tensor<1x216x288x24xf32>
  return %0 : tensor<1x216x288x24xf32>
}
}