module {
func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  %1 = tfl.mul %0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
}