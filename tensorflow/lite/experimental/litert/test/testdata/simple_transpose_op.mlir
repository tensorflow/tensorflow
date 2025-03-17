module {
func.func @main(%arg0: tensor<128x4x2x128xf32>) -> tensor<128x2x4x128xf32> {
  %cst = "tfl.pseudo_const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<128x4x2x128xf32>, tensor<4xi32>) -> tensor<128x2x4x128xf32>
  return %0 : tensor<128x2x4x128xf32>
}
}