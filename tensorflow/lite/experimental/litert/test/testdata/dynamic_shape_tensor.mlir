module {
func.func @main(%arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}
}