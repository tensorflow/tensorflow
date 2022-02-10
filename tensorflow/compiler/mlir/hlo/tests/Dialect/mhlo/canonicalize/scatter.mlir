// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.func(canonicalize)' | FileCheck %s

// Folding this case would explode the IR
func @scatter_fold_explosion() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<1> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<511x1x6400x6400xf32>
  // CHECK: mhlo.scatter
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [3]>, unique_indices = true} : (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<511x1x6400x6400xf32>) -> tensor<512x1x6400x6400xf32>

  return %scatter :  tensor<512x1x6400x6400xf32>
}
