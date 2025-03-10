module {
func.func @main(%arg0: tensor<1x1xf32>) -> tensor<f32> {
  %cst = arith.constant dense<[]> : tensor<0xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<1x1xf32>, tensor<0xi32>) -> tensor<f32>
  return %0 : tensor<f32>
}
}