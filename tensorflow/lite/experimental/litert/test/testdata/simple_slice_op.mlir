module {
func.func @main(%arg0: tensor<1x128x8x256xf32>) -> tensor<1x128x8x128xf32> {
  %cst_0 = "tfl.pseudo_const"() <{value = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
  %cst_1 = "tfl.pseudo_const"() <{value = dense<[1, 128, 4, 128]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %0 = "tfl.slice"(%arg0, %cst_0, %cst_1) : (tensor<1x128x8x256xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x128x8x128xf32>
  return %0 : tensor<1x128x8x128xf32>
}
}