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
// RUN: odml-to-stablehlo-opt %s -stablehlo-fold-broadcast -cse -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_float
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_float() -> (tensor<1x1x2x4xf32>) {
  // CHECK-DAG: %[[RES:.*]] = stablehlo.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00, 9.000000e+00, 1.600000e+01], [5.000000e+00, 1.200000e+01, 2.100000e+01, 3.200000e+01]]]]> : tensor<1x1x2x4xf32>
  %cst0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst1 = stablehlo.constant dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]> : tensor<1x1x2x4xf32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xf32>) -> tensor<1x1x2x4xf32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x1x2x4xf32>
  // CHECK:      return %[[RES]] : tensor<1x1x2x4xf32>
  func.return %1 : tensor<1x1x2x4xf32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_2D_float
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_2D_float() -> (tensor<1x2x2x3xf32>) {
  // CHECK-DAG: %[[RES:.*]] = stablehlo.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00, 9.000000e+00], [4.000000e+00, 1.000000e+01, 1.800000e+01]], {{\[\[}}2.800000e+01, 4.000000e+01, 5.400000e+01], [4.000000e+01, 5.500000e+01, 7.200000e+01]]]]> : tensor<1x2x2x3xf32>
  %cst0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %cst1 = stablehlo.constant dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [1, 3] : (tensor<2x3xf32>) -> tensor<1x2x2x3xf32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x2x2x3xf32>
  // CHECK:      return %[[RES]] : tensor<1x2x2x3xf32>
  func.return %1 : tensor<1x2x2x3xf32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_int
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_int() -> (tensor<1x1x2x4xi32>) {
  // CHECK-DAG: %[[RES:.*]] = stablehlo.constant dense<{{\[\[\[\[}}1, 4, 9, 16], [5, 12, 21, 32]]]]> : tensor<1x1x2x4xi32>
  %cst0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %cst1 = stablehlo.constant dense<[[[[1, 2, 3, 4], [5, 6, 7, 8]]]]> : tensor<1x1x2x4xi32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x1x2x4xi32>
  // CHECK:      return %[[RES]] : tensor<1x1x2x4xi32>
  func.return %1 : tensor<1x1x2x4xi32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_4D_int
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_4D_int() -> tensor<1x2x1x4xi32> {
  // CHECK-DAG: %[[RES:.*]] = stablehlo.constant dense<{{\[\[\[\[}}0, 1, 4, 9]], {{\[\[}}0, 1, 4, 9]]]]> : tensor<1x2x1x4xi32>
  %0 = stablehlo.constant dense<[[[[0, 1, 2, 3]]]]> : tensor<1x1x1x4xi32>
  %1 = stablehlo.constant dense<[[[[0, 1, 2, 3]], [[0, 1, 2, 3]]]]> : tensor<1x2x1x4xi32>
  %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1x1x4xi32>) -> tensor<1x2x1x4xi32>
  %3 = stablehlo.multiply %1, %2 : tensor<1x2x1x4xi32>
  // CHECK:      return %[[RES]] : tensor<1x2x1x4xi32>
  return %3 : tensor<1x2x1x4xi32>
}

// CHECK-LABEL: @notFoldBroadcastInDimBeforeMulOpWhenArgIsNonConst_bcast_dim_1D_int
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi32>
func.func @notFoldBroadcastInDimBeforeMulOpWhenArgIsNonConst_bcast_dim_1D_int(%arg0: tensor<1x1x2x4xi32>) -> (tensor<1x1x2x4xi32>) {
  // CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<{{\[}}1, 2, 3, 4]> : tensor<4xi32>
  %cst0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CONST]], dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[BROADCAST]], %[[ARG]] : tensor<1x1x2x4xi32>
  %1 = stablehlo.multiply %0, %arg0 : tensor<1x1x2x4xi32>
  // CHECK:      return %[[MUL]] : tensor<1x1x2x4xi32>
  func.return %1 : tensor<1x1x2x4xi32>
}

// CHECK-LABEL: @foldSplatBroadcastInDimBeforeMulOp
func.func @foldSplatBroadcastInDimBeforeMulOp() -> tensor<1x1x2x4xi32> {
  // CHECK: %[[RES:.*]] = stablehlo.constant dense<6> : tensor<1x1x2x4xi32>
  %cst0 = stablehlo.constant dense<2> : tensor<4xi32>
  %cst1 = stablehlo.constant dense<3> : tensor<1x1x2x4xi32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x1x2x4xi32>
  // CHECK: return %[[RES]] : tensor<1x1x2x4xi32>
  func.return %1 : tensor<1x1x2x4xi32>
}

// Only the multiply with a constant operand gets the folded constant; the
// multiply with a non-constant operand keeps the broadcast.
// CHECK-LABEL: @foldOnlyConstMulUseOfMultiUseBroadcastInDim
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x1x2x4xi32>)
func.func @foldOnlyConstMulUseOfMultiUseBroadcastInDim(%arg0: tensor<1x1x2x4xi32>) -> (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) {
  // CHECK-DAG: %[[RES:.*]] = stablehlo.constant dense<{{\[\[\[\[}}1, 4, 9, 16], [5, 12, 21, 32]]]]> : tensor<1x1x2x4xi32>
  // CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<{{\[}}1, 2, 3, 4]> : tensor<4xi32>
  %cst0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %cst1 = stablehlo.constant dense<[[[[1, 2, 3, 4], [5, 6, 7, 8]]]]> : tensor<1x1x2x4xi32>
  // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CONST]], dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x1x2x4xi32>
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[BROADCAST]], %[[ARG]] : tensor<1x1x2x4xi32>
  %2 = stablehlo.multiply %0, %arg0 : tensor<1x1x2x4xi32>
  // CHECK: return %[[RES]], %[[MUL]]
  func.return %1, %2 : tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>
}

// The other multiply operand is not a DenseElementsAttr, so neither the
// broadcast nor the multiply can be folded.
// CHECK-LABEL: @notFoldBroadcastInDimBeforeMulOpWithResourceConst
func.func @notFoldBroadcastInDimBeforeMulOpWithResourceConst() -> tensor<1x1x2x4xf32> {
  // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[BROADCAST]]
  %cst0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst1 = stablehlo.constant dense_resource<ones_blob> : tensor<1x1x2x4xf32>
  %0 = stablehlo.broadcast_in_dim %cst0, dims = [3] : (tensor<4xf32>) -> tensor<1x1x2x4xf32>
  %1 = stablehlo.multiply %0, %cst1 : tensor<1x1x2x4xf32>
  // CHECK: return %[[MUL]]
  func.return %1 : tensor<1x1x2x4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      ones_blob: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}
