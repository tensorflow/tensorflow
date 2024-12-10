module {
func.func @main(%arg0: tensor<8x1x1x100xi1>, %arg1: tensor<8x100x32x100xf32>, %arg2: tensor<8x100x32x100xf32>) -> tensor<8x100x32x100xf32> {
  %0 = "tfl.select_v2"(%arg0, %arg1, %arg2) : (tensor<8x1x1x100xi1>, tensor<8x100x32x100xf32>, tensor<8x100x32x100xf32>) -> tensor<8x100x32x100xf32>
  return %0 : tensor<8x100x32x100xf32>
}
}