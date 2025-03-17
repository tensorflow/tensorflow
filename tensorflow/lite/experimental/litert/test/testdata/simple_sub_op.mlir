module {
func.func @main(%arg0: tensor<1x128x4x128xf32>, %arg1: tensor<1x128x4x128xf32>) -> tensor<1x128x4x128xf32> {
  %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128x4x128xf32>
  return %0 : tensor<1x128x4x128xf32>
}
}