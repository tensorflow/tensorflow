module {
func.func @main(%arg0: tensor<8x100x1xi32>) -> tensor<8x100x1xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<8x100x1xi32>) -> tensor<8x100x1xf32>
  return %0 : tensor<8x100x1xf32>
}
}
