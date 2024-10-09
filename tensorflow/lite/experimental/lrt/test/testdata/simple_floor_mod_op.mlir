module {
func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = "tfl.floor_mod"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
}