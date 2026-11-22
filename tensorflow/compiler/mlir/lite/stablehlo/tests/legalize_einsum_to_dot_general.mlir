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
// RUN: odml-to-stablehlo-opt %s -stablehlo-legalize-einsum-to-dot-general -split-input-file | FileCheck %s

// NOTE: 2-operand diagonal not supported as a dot_general lowering.
// CHECK-LABEL: func @einsum_diag
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         %[[CST:.+]] = stablehlo.constant dense<{{.*}} : tensor<f32>
// CHECK:         stablehlo.einsum %[[CST]], %[[ARG0]], config = ",ii->i"
// CHECK-NOT:     stablehlo.dot_general
func.func @einsum_diag(%arg0: tensor<6x6xf32>) -> tensor<6xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "stablehlo.einsum"(%0, %arg0) <{einsum_config = ",ii->i"}> : (tensor<f32>, tensor<6x6xf32>) -> tensor<6xf32>
  func.return %1 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func @einsum_batched_matrix_high_rank_vector_mul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[DG:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0] x [0]
// CHECK-SAME:      contracting_dims = [2] x [3]
// CHECK-SAME:    : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x2x5x3xf32>
// CHECK:         %[[T:.+]] = stablehlo.transpose %[[DG]]
// CHECK-SAME:      dims = [0, 2, 3, 1]
// CHECK-SAME:    : (tensor<8x2x5x3xf32>) -> tensor<8x5x3x2xf32>
// CHECK:         return %[[T]]
func.func @einsum_batched_matrix_high_rank_vector_mul(%arg0: tensor<8x2x6xf32>, %arg1: tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) <{einsum_config = "bxy,bijy->bijx"}> : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32>
  func.return %0 : tensor<8x5x3x2xf32>
}

// -----

// Batched matmul already in natural dot_general result order: no transpose.
// CHECK-LABEL: func @einsum_batch_matmul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[DG:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0] x [0]
// CHECK-SAME:      contracting_dims = [2] x [1]
// CHECK-SAME:    : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
// CHECK-NOT:     stablehlo.transpose
// CHECK:         return %[[DG]]
func.func @einsum_batch_matmul(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "bij,bjk->bik" : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @matmul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [1] x [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) <{einsum_config = "ij,jk->ik"}> : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @matvec
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [1] x [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
func.func @matvec(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) <{einsum_config = "ij,j->i"}> : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dot
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [0] x [0]
// CHECK-SAME:    : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
func.func @dot(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) <{einsum_config = "i,i->"}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Contracting dims are paired by label, not by position.
// CHECK-LABEL: func @transposed_dot_square
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [0, 1] x [1, 0]
// CHECK-SAME:    : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<f32>
func.func @transposed_dot_square(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<f32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "ij,ji->" : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @transposed_dot_non_square
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [0, 1] x [1, 0]
// CHECK-SAME:    : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<f32>
func.func @transposed_dot_non_square(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<f32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "ij,ji->" : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Batching dims are paired by label, not by position.
// CHECK-LABEL: func @batch_labels_in_different_order
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[DG:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0, 1] x [1, 0]
// CHECK-SAME:    : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NOT:     stablehlo.transpose
// CHECK:         return %[[DG]]
func.func @batch_labels_in_different_order(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "ab,ba->ab" : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @batch_matmul_rhs_batch_not_leading
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[DG:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0] x [1]
// CHECK-SAME:      contracting_dims = [2] x [0]
// CHECK-SAME:    : (tensor<2x3x4xf32>, tensor<4x2x5xf32>) -> tensor<2x3x5xf32>
// CHECK:         return %[[DG]]
func.func @batch_matmul_rhs_batch_not_leading(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4x2x5xf32>) -> tensor<2x3x5xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "bij,jbk->bik" : (tensor<2x3x4xf32>, tensor<4x2x5xf32>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// Binary form of a unary einsum (as produced from TF): the rhs labels are
// reduced in one operand only, so this is not a dot_general.
// CHECK-LABEL: func @unary_einsum_full_reduction
// CHECK:         stablehlo.einsum
// CHECK-NOT:     stablehlo.dot_general
func.func @unary_einsum_full_reduction(%arg0: tensor<2x3xf32>) -> tensor<f32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.einsum %0, %arg0, config = ",ij->" : (tensor<f32>, tensor<2x3xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: func @unary_einsum_partial_reduction
// CHECK:         stablehlo.einsum
// CHECK-NOT:     stablehlo.dot_general
func.func @unary_einsum_partial_reduction(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.einsum %0, %arg0, config = ",ij->i" : (tensor<f32>, tensor<2x3xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @lhs_only_reduction
// CHECK:         stablehlo.einsum
// CHECK-NOT:     stablehlo.dot_general
func.func @lhs_only_reduction(%arg0: tensor<2x3xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "ij,k->k" : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @repeated_label_in_operand
// CHECK:         stablehlo.einsum
// CHECK-NOT:     stablehlo.dot_general
func.func @repeated_label_in_operand(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "ii,i->i" : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @ellipsis
// CHECK:         stablehlo.einsum
// CHECK-NOT:     stablehlo.dot_general
func.func @ellipsis(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = stablehlo.einsum %arg0, %arg1, config = "...ij,...jk->...ik" : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}
