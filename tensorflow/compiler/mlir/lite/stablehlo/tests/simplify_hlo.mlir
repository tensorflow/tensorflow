// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: odml-to-stablehlo-opt %s -simplify-hlo -split-input-file | FileCheck %s --dump-input=fail

// Integer constant arithmetic is folded.

// CHECK-LABEL: fold_integer_add
func.func @fold_integer_add() -> tensor<2xi32> {
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi32>
  %2 = stablehlo.add %0, %1 : tensor<2xi32>
  func.return %2 : tensor<2xi32>
}

// CHECK:     %[[CST:.*]] = stablehlo.constant dense<[4, 6]> : tensor<2xi32>
// CHECK-NOT: stablehlo.add
// CHECK:     return %[[CST]]

// -----

// Float constant arithmetic is folded too, as MHLO's folders did.

// CHECK-LABEL: fold_float_add
func.func @fold_float_add() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2xf32>
  func.return %2 : tensor<2xf32>
}

// CHECK:     %[[CST:.*]] = stablehlo.constant dense<[4.000000e+00, 6.000000e+00]> : tensor<2xf32>
// CHECK-NOT: stablehlo.add
// CHECK:     return %[[CST]]

// -----

// JAX's floor_divide by a constant: the `x2 == 0` select, `sign(x2)` and the
// broadcasts of the constants are folded, leaving the form the floor_div
// legalization patterns match.

// CHECK-LABEL: fold_floor_div_by_constant
func.func @fold_floor_div_by_constant(%arg0: tensor<12xf32>) -> tensor<12xf32> {
  %c2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %two = stablehlo.broadcast_in_dim %c2, dims = [] : (tensor<f32>) -> tensor<12xf32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %zero = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<f32>) -> tensor<12xf32>
  %c1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %one = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<f32>) -> tensor<12xf32>
  %mod = stablehlo.remainder %arg0, %two : tensor<12xf32>
  %is_zero = stablehlo.compare EQ, %two, %zero, FLOAT : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xi1>
  %sub = stablehlo.subtract %arg0, %mod : tensor<12xf32>
  %x1 = stablehlo.select %is_zero, %arg0, %sub : tensor<12xi1>, tensor<12xf32>
  %div = stablehlo.divide %x1, %two : tensor<12xf32>
  %mod_ne_0 = stablehlo.compare NE, %mod, %zero, FLOAT : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xi1>
  %sign_two = stablehlo.sign %two : tensor<12xf32>
  %sign_mod = stablehlo.sign %mod : tensor<12xf32>
  %signs_ne = stablehlo.compare NE, %sign_two, %sign_mod, FLOAT : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xi1>
  %ind = stablehlo.and %mod_ne_0, %signs_ne : tensor<12xi1>
  %div_m1 = stablehlo.subtract %div, %one : tensor<12xf32>
  %sel = stablehlo.select %ind, %div_m1, %div : tensor<12xi1>, tensor<12xf32>
  %r = stablehlo.round_nearest_afz %sel : tensor<12xf32>
  func.return %r : tensor<12xf32>
}

// CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<12xf32>
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<12xf32>
// CHECK-DAG: %[[TWO:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<12xf32>
// CHECK:     %[[MOD:.*]] = stablehlo.remainder %arg0, %[[TWO]]
// CHECK:     %[[SUB:.*]] = stablehlo.subtract %arg0, %[[MOD]]
// CHECK:     %[[DIV:.*]] = stablehlo.divide %[[SUB]], %[[TWO]]
// CHECK:     stablehlo.compare NE, %[[MOD]], %[[ZERO]]
// CHECK:     %[[SIGN_MOD:.*]] = stablehlo.sign %[[MOD]]
// CHECK:     stablehlo.compare NE, %[[ONE]], %[[SIGN_MOD]]
// CHECK-NOT: stablehlo.broadcast_in_dim
// CHECK:     stablehlo.round_nearest_afz

// -----

// transpose(broadcast_in_dim(x)) is merged into a single broadcast_in_dim
// (instead of becoming a reshape), so it can merge with a following
// broadcast_in_dim.

// CHECK-LABEL: merge_transpose_of_broadcast_in_dim
func.func @merge_transpose_of_broadcast_in_dim(%arg0: tensor<1x1x1x36xi1>) -> tensor<1x2x36x36xi1> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x1x36xi1>) -> tensor<1x36x1x36xi1>
  %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1x36x1x36xi1>) -> tensor<1x1x36x36xi1>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x36x36xi1>) -> tensor<1x2x36x36xi1>
  func.return %2 : tensor<1x2x36x36xi1>
}

// CHECK:     %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x1x36xi1>) -> tensor<1x2x36x36xi1>
// CHECK-NOT: stablehlo.transpose
// CHECK-NOT: stablehlo.reshape
// CHECK:     return %[[BCAST]]

// -----

// When the transpose moves a unit dimension of the broadcast operand past a
// non-unit one, the operand is reshaped so that the merged broadcast_in_dim
// keeps increasing broadcast dimensions (which the TFL legalization requires).

// CHECK-LABEL: merge_transpose_of_broadcast_in_dim_with_reshape
func.func @merge_transpose_of_broadcast_in_dim_with_reshape(%arg0: tensor<1x5xf32>) -> tensor<5x3xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x5xf32>) -> tensor<3x5xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
  func.return %1 : tensor<5x3xf32>
}

// CHECK:     %[[RESHAPE:.*]] = stablehlo.reshape %arg0 : (tensor<1x5xf32>) -> tensor<5x1xf32>
// CHECK:     %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[RESHAPE]], dims = [0, 1] : (tensor<5x1xf32>) -> tensor<5x3xf32>
// CHECK-NOT: stablehlo.transpose
// CHECK:     return %[[BCAST]]

// -----

// A transpose that reorders non-unit dimensions of the broadcast operand is
// not merged: the result would need non-increasing broadcast dimensions.

// CHECK-LABEL: no_merge_transpose_of_broadcast_in_dim_reordering_non_unit_dims
func.func @no_merge_transpose_of_broadcast_in_dim_reordering_non_unit_dims(%arg0: tensor<8x16xf32>) -> tensor<2x16x8xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<2x8x16xf32>
  %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<2x8x16xf32>) -> tensor<2x16x8xf32>
  func.return %1 : tensor<2x16x8xf32>
}

// CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<2x8x16xf32>
// CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[BCAST]], dims = [0, 2, 1]
// CHECK: return %[[TRANSPOSE]]

// -----

// Identity transposes are removed by the canonicalization patterns.

// CHECK-LABEL: remove_identity_transpose
func.func @remove_identity_transpose(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// CHECK-NOT: stablehlo.transpose
// CHECK:     return %arg0

// -----

// StableHLO's constant transpose folder is excluded: transposes of non-splat
// constants are not materialized.

// CHECK-LABEL: no_fold_non_splat_constant_transpose
func.func @no_fold_non_splat_constant_transpose() -> tensor<3x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}

// CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose
// CHECK: return %[[TRANSPOSE]]

// -----

// Same for integer constants.

// CHECK-LABEL: no_fold_non_splat_integer_constant_transpose
func.func @no_fold_non_splat_integer_constant_transpose() -> tensor<3x2xi32> {
  %0 = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}

// CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose
// CHECK: return %[[TRANSPOSE]]

// -----

// Adjacent identical splat operands of `concatenate` are not merged (StableHLO's
// FoldConcatenateAdjacentSplatsOpPattern is excluded).

// CHECK-LABEL: no_merge_concatenate_adjacent_splats
func.func @no_merge_concatenate_adjacent_splats(%arg0: tensor<2x3xf32>) -> tensor<6x3xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2x3xf32>
  %1 = stablehlo.concatenate %0, %0, %arg0, dim = 0 : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2x3xf32>
// CHECK: stablehlo.concatenate %[[CST]], %[[CST]], %arg0, dim = 0
