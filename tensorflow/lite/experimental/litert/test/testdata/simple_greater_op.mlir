module {
func.func @main(%arg0: tensor<1x1x64xi32>, %arg1: tensor<1x1x64xi32>) -> tensor<1x1x64xi1> {
  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<1x1x64xi32>, tensor<1x1x64xi32>) -> tensor<1x1x64xi1>
  return %0 : tensor<1x1x64xi1>
}
}