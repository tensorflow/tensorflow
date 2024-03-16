// RUN: stablehlo-quant-opt %s -stablehlo-defer-activation-transpose \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// Tests that an `add(transpose(arg0), arg1)` pattern is converted to
// `transpose(add(arg0, transpose(arg1)))`. The transpose in the activation is
// deferred to the output of `stablehlo.add` and an extra transpose op is
// inserted to the RHS to match the shape of the operand.

// CHECK-LABEL: add_with_activation_transpose
func.func @add_with_activation_transpose(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<1x4x3x3xf32>
  %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
  %2 = stablehlo.add %1, %0 : tensor<1x4x3x3xf32>
  return %2 : tensor<1x4x3x3xf32>
}
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[CONST_0]], dims = [0, 2, 3, 1] : (tensor<1x4x3x3xf32>) -> tensor<1x3x3x4xf32>

// Check that the shape of the add is changed to reflect the deferred transpose.
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[ARG_0]], %[[TRANSPOSE_0]] : tensor<1x3x3x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose
// CHECK: return %[[TRANSPOSE_1]]

// -----

// [No change] Tests that the activation transpose whose permutation is not
// `[0, 3, 1, 2]` is not deferred.

// CHECK-LABEL: add_with_activation_transpose_permutation_mismatch
func.func @add_with_activation_transpose_permutation_mismatch(
      %arg0: tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<1x3x2x4xf32>
  %1 = stablehlo.transpose %arg0, dims = [0, 2, 1, 3] : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
  %2 = stablehlo.add %1, %0 : tensor<1x3x2x4xf32>
  return %2 : tensor<1x3x2x4xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[ADD_0]]

// -----

// [No change] Tests that the activation transpose whose rank is not 4 is not
// deferred.

// CHECK-LABEL: add_with_activation_transpose_rank_two
func.func @add_with_activation_transpose_rank_two(%arg0: tensor<1x2xf32>) -> tensor<2x1xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<2x1xf32>
  %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %2 = stablehlo.add %1, %0 : tensor<2x1xf32>
  return %2 : tensor<2x1xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[ADD_0]]

// -----

// [No change] Tests that the right-hand side that is not a constant is not
// deferred.

// CHECK-LABEL: add_with_activation_transpose_nonconst_rhs
func.func @add_with_activation_transpose_nonconst_rhs(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<1x4x3x3xf32>
  return %1 : tensor<1x4x3x3xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[ADD_0]]
