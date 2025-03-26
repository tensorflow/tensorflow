module {

func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[-1.0, -1.0, -1.0, -1.0]> : tensor<4xf32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @func1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @func2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

}