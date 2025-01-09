module {

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @func1(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

}