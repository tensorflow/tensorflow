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

// Float arithmetic is not rewritten (`optimizeFloat` is disabled).

// CHECK-LABEL: no_fold_float_add
func.func @no_fold_float_add() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2xf32>
  func.return %2 : tensor<2xf32>
}

// CHECK: %[[ADD:.*]] = stablehlo.add
// CHECK: return %[[ADD]]

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
