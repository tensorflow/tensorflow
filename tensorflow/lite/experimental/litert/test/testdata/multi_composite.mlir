func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.composite "odml.npu_call" %arg0, %arg1 {decomposition = @decomp1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.composite "odml.regular_composite" %arg0, %0 {decomposition = @decomp2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.composite "odml.npu_call" %arg0, %1 {decomposition = @decomp3} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

func.func private @decomp1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func private @decomp2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func private @decomp3(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
