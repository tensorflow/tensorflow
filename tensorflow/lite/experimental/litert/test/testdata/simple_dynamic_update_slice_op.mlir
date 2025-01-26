module {
func.func @main(%arg0: tensor<1x64x4x64xf32>, %arg1: tensor<1x1x4x64xf32>) -> tensor<1x64x4x64xf32> {
  %cst = "tfl.pseudo_const"() <{value = dense<[0, 1, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %0 = "tfl.dynamic_update_slice"(%arg0, %arg1, %cst) : (tensor<1x64x4x64xf32>, tensor<1x1x4x64xf32>, tensor<4xi32>) -> tensor<1x64x4x64xf32>
  return %0 : tensor<1x64x4x64xf32>
}
}
