module {
func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4xi32> {
  // %cst = "tfl.pseudo_const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tfl.pack"(%arg0, %arg1, %arg2, %arg3) <{axis = 0 : i32, values_count = 4 : i32}> : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
}
