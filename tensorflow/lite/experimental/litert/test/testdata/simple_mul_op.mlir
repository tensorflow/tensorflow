module {
func.func @main(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<1x128x2304xf32>) -> tensor<1x128x2304xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128x2304xf32>
  return %0 : tensor<1x128x2304xf32>
}
}