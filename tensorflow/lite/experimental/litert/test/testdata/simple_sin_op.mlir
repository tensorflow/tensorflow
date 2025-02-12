module {
func.func @main(%arg0: tensor<8x100x1x2xf32>) -> tensor<8x100x1x2xf32> {
  %0 = "tfl.sin"(%arg0) : (tensor<8x100x1x2xf32>) -> tensor<8x100x1x2xf32>
  return %0 : tensor<8x100x1x2xf32>
}
}