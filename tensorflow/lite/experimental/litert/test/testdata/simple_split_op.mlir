module {
func.func @main(%arg0: tensor<1x4x3x3xf32>) -> tensor<1x4x3x1xf32> {
  %cst = "tfl.pseudo_const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  %0:3 = "tfl.split"(%cst, %arg0) <{num_splits = 3 : i32}> : (tensor<i32>, tensor<1x4x3x3xf32>) -> (tensor<1x4x3x1xf32>, tensor<1x4x3x1xf32>, tensor<1x4x3x1xf32>)
  return %0#0 : tensor<1x4x3x1xf32>
}
}