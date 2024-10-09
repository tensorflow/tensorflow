module {
func.func @main(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<131072x4xi32>, %arg2: tensor<131072xf32>) -> tensor<1x128x4x256xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1x128x4x256xf32>, tensor<131072x4xi32>, tensor<131072xf32>) -> tensor<1x128x4x256xf32>
  return %0 : tensor<1x128x4x256xf32>
}
}