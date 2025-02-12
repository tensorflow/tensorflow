module {
func.func @main(%arg0: tensor<2x3x6xf32>, %arg1: tensor<4x5xi32>) -> tensor<4x5x3x6xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<2x3x6xf32>, tensor<4x5xi32>) -> tensor<4x5x3x6xf32>
  return %0 : tensor<4x5x3x6xf32>
}
}