module {
func.func @main(%arg0: tensor<1x128x2304xf32>) -> tensor<1x128x1xf32> {
  %cst = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst) <{keep_dims = true}> : (tensor<1x128x2304xf32>, tensor<1xi32>) -> tensor<1x128x1xf32>
  return %0 : tensor<1x128x1xf32>
}
}
