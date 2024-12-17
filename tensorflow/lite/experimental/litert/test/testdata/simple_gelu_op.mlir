module {
func.func @main(%arg0: tensor<8x100x1xf32>) -> tensor<8x100x1xf32> {
  %0 = "tfl.gelu"(%arg0) : (tensor<8x100x1xf32>) -> tensor<8x100x1xf32>
  return %0 : tensor<8x100x1xf32>
}
}
