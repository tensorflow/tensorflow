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

// Tests that an `add(transpose(arg0), broadcast_in_dim(arg1))` pattern is
// converted to `transpose(add(arg0, transpose(broadcast_in_dim(arg1))))`.
// The transpose in the activation is deferred to the output of `stablehlo.add`
// and an extra transpose op is inserted to the RHS to match the shape of the
// operand.

// CHECK-LABEL: add_with_activation_transpose_broadcasted_rhs
func.func @add_with_activation_transpose_broadcasted_rhs(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<1x4x3x3xf32>
  %2 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
  %3 = stablehlo.add %2, %1 : tensor<1x4x3x3xf32>
  return %3 : tensor<1x4x3x3xf32>
}
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant
// CHECK-DAG: %[[BROADCAST:.+]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [1] : (tensor<4xf32>) -> tensor<1x4x3x3xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[BROADCAST]], dims = [0, 2, 3, 1] : (tensor<1x4x3x3xf32>) -> tensor<1x3x3x4xf32>

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

// -----

// Tests that the transpose of the input of `stablehlo.reduce_window` is
// deferred to the result. The attributes are permutated according to the new
// input shape.

// CHECK-LABEL: reduce_window_max_activation_transpose
func.func @reduce_window_max_activation_transpose(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x4x8x8xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x16x16x4xf32>) -> tensor<1x4x16x16xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x8x8xf32>
  return %2 : tensor<1x4x8x8xf32>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<1x16x16x4xf32>
// CHECK-DAG: %[[INIT_VALUE_CONST:.+]] = stablehlo.constant dense<0xFF800000>

// Check that the body is not modified.
// CHECK: %[[REDUCE_WINDOW:.+]] = "stablehlo.reduce_window"(%[[ARG]], %[[INIT_VALUE_CONST]])
// CHECK: ^bb0(%[[REDUCE_ARG_0:.+]]: tensor<f32>, %[[REDUCE_ARG_1:.+]]: tensor<f32>):
// CHECK: %[[MAX:.+]] = stablehlo.maximum %[[REDUCE_ARG_0]], %[[REDUCE_ARG_1]]
// CHECK: stablehlo.return %[[MAX]]

// Check that the attributes window_dimensions & window_strides are also
// permutated to match the new input shape.
// CHECK: {window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}
// CHECK-SAME: (tensor<1x16x16x4xf32>, tensor<f32>) -> tensor<1x8x8x4xf32>

// Check that a `stablehlo.transpose` is added to the result to match the shape
// of the users.
// CHECK: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[REDUCE_WINDOW]], dims = [0, 3, 1, 2] : (tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32>
// CHECK: return %[[TRANSPOSE]]

// -----

// Tests that the transpose of the input of `stablehlo.reduce_window` is
// deferred to the result. The attributes are permutated according to the new
// input shape. This test is similar to the test above with the difference that
// the `stablehlo.reduce_window` has explicit optional attributes:
// `base_dilations` and `window_dilations`.

// CHECK-LABEL: reduce_window_max_activation_transpose_explicit_optional_attrs
func.func @reduce_window_max_activation_transpose_explicit_optional_attrs(
      %arg0: tensor<1x16x16x4xf32>) -> tensor<1x4x15x15xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x16x16x4xf32>) -> tensor<1x4x16x16xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 2, 2>,
    base_dilations = array<i64: 1, 1, 2, 2>,
    window_dilations = array<i64: 1, 1, 2, 2>
  } : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x15x15xf32>
  return %2 : tensor<1x4x15x15xf32>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<1x16x16x4xf32>
// CHECK-DAG: %[[INIT_VALUE_CONST:.+]] = stablehlo.constant dense<0xFF800000>

// Check that the body is not modified.
// CHECK: %[[REDUCE_WINDOW:.+]] = "stablehlo.reduce_window"(%[[ARG]], %[[INIT_VALUE_CONST]])
// CHECK: ^bb0(%[[REDUCE_ARG_0:.+]]: tensor<f32>, %[[REDUCE_ARG_1:.+]]: tensor<f32>):
// CHECK: %[[MAX:.+]] = stablehlo.maximum %[[REDUCE_ARG_0]], %[[REDUCE_ARG_1]]
// CHECK: stablehlo.return %[[MAX]]

// Check that the attributes window_dimensions & window_strides along with
// optional attributes base_dilations and window_dilations are also permutated
// to match the new input shape.
// CHECK: {base_dilations = array<i64: 1, 2, 2, 1>, window_dilations = array<i64: 1, 2, 2, 1>, window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}
// CHECK-SAME: (tensor<1x16x16x4xf32>, tensor<f32>) -> tensor<1x15x15x4xf32>

// Check that a `stablehlo.transpose` is added to the result to match the shape
// of the users.
// CHECK: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[REDUCE_WINDOW]], dims = [0, 3, 1, 2] : (tensor<1x15x15x4xf32>) -> tensor<1x4x15x15xf32>
// CHECK: return %[[TRANSPOSE]]

// -----

// [No change] Tests that the transpose of the input of
// `stablehlo.reduce_window` is NOT deferred to the result, when the input
// tensor does not have rank 4.

// CHECK-LABEL: reduce_window_max_activation_transpose
// CHECK-SAME: (%[[ARG:.+]]: tensor<16x8xf32>) -> tensor<4x8xf32>
func.func @reduce_window_max_activation_transpose_rank2(%arg0: tensor<16x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<16x8xf32>) -> tensor<8x16xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 2>} : (tensor<8x16xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}
// CHECK-DAG: stablehlo.constant
// CHECK: stablehlo.transpose %[[ARG]]
// CHECK: stablehlo.reduce_window

// -----

// [No change] Tests that the transpose of the input of
// `stablehlo.reduce_window` is NOT deferred to the result, when it has an
// explicit `padding` attribute.

// CHECK-LABEL: reduce_window_max_activation_transpose_with_padding
func.func @reduce_window_max_activation_transpose_with_padding(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x4x9x9xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x16x16x4xf32>) -> tensor<1x4x16x16xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 2, 2>,
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>
  } : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x9x9xf32>
  return %2 : tensor<1x4x9x9xf32>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<1x16x16x4xf32>
// CHECK-DAG: stablehlo.constant
// CHECK: stablehlo.transpose %[[ARG]]
// CHECK: stablehlo.reduce_window

// -----

// [No change] Tests that the transpose of the input of
// `stablehlo.reduce_window` is NOT deferred to the result, when the transpose
// isn't `[0, 3, 1, 2]` (i.e. NCHW->NHWC).

// CHECK-LABEL: reduce_window_max_activation_transpose_with_padding
func.func @reduce_window_max_activation_transpose_with_padding(%arg0: tensor<16x16x4x1xf32>) -> tensor<1x4x8x8xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %1 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<16x16x4x1xf32>) -> tensor<1x4x16x16xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 2, 2>
  } : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x8x8xf32>
  return %2 : tensor<1x4x8x8xf32>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<16x16x4x1xf32>
// CHECK-DAG: stablehlo.constant
// CHECK: stablehlo.transpose %[[ARG]]
// CHECK: stablehlo.reduce_window

// -----

// Tests that an `max(transpose(arg0), arg1)` pattern is converted to
// `transpose(max(arg0, transpose(arg1)))`. The transpose in the activation is
// deferred to the output of `stablehlo.max` and an extra transpose op is
// inserted to the RHS to match the shape of the operand.

// CHECK-LABEL: max_with_activation_transpose
func.func @max_with_activation_transpose(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<1x4x3x3xf32>
  %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x4x3x3xf32>
  return %2 : tensor<1x4x3x3xf32>
}
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[CONST_0]], dims = [0, 2, 3, 1] : (tensor<1x4x3x3xf32>) -> tensor<1x3x3x4xf32>

// Check that the shape of the add is changed to reflect the deferred transpose.
// CHECK: %[[MAX_0:.+]] = stablehlo.maximum %[[ARG_0]], %[[TRANSPOSE_0]] : tensor<1x3x3x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose
// CHECK: return %[[TRANSPOSE_1]]

// -----

// [No change] Tests that the activation transpose of `stablehlo.maximum` whose
// permutation is not `[0, 3, 1, 2]` is not deferred.

// CHECK-LABEL: max_with_activation_transpose_permutation_mismatch
func.func @max_with_activation_transpose_permutation_mismatch(
      %arg0: tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<1x3x2x4xf32>
  %1 = stablehlo.transpose %arg0, dims = [0, 2, 1, 3] : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x3x2x4xf32>
  return %2 : tensor<1x3x2x4xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[MAX_0:.+]] = stablehlo.maximum %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[MAX_0]]

// -----

// [No change] Tests that the activation transpose of `stablehlo.maximum` whose
// rank is not 4 is not deferred.

// CHECK-LABEL: max_with_activation_transpose_rank_two
func.func @max_with_activation_transpose_rank_two(%arg0: tensor<1x2xf32>) -> tensor<2x1xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<2x1xf32>
  %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<2x1xf32>
  return %2 : tensor<2x1xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[MAX_0:.+]] = stablehlo.maximum %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[MAX_0]]
