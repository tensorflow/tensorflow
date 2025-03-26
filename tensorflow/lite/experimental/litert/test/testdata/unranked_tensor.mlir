module {
func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0 : tensor<*xf32>
}
}