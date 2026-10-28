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
