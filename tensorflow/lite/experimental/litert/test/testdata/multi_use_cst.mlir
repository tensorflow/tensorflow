module {
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<4xf32>
  %2 = tfl.add %1, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %2 : tensor<4xf32>
}
}