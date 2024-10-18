module {
func.func @main(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %0 = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
  return %0 : tensor<8x128xf32>
}
}