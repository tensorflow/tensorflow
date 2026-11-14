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
// RUN: odml-to-stablehlo-opt %s -stablehlo-legalize-torch-index-select-to-gather -split-input-file | FileCheck %s

// Ported from MHLO's hlo-legalize-torch-index-select-to-gather.mlir.

// CHECK-LABEL: func @index_select_to_gather_convert_index_type
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[IDX:.+]] = stablehlo.convert %[[ARG1]] : (tensor<2xi64>) -> tensor<2xui32>
// CHECK:         %[[RES:.+]] = "stablehlo.gather"(%[[ARG0]], %[[IDX]])
// CHECK-SAME:      offset_dims = [1, 2]
// CHECK-SAME:      collapsed_slice_dims = [0]
// CHECK-SAME:      start_index_map = [0]
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      slice_sizes = array<i64: 1, 1, 5>
// CHECK-SAME:    : (tensor<5x1x5xi64>, tensor<2xui32>) -> tensor<2x1x5xi64>
// CHECK:         return %[[RES]]
func.func @index_select_to_gather_convert_index_type(%arg0 : tensor<5x1x5xi64>, %arg1 : tensor<2xi64>) -> tensor<2x1x5xi64> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x1x5xi64>, tensor<2xi64>) -> tensor<2x1x5xi64>
  func.return %0 : tensor<2x1x5xi64>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_multi_offset_dims
// CHECK:         "stablehlo.gather"(%arg0, %arg1)
// CHECK-SAME:      offset_dims = [1, 2]
// CHECK-SAME:      collapsed_slice_dims = [0]
// CHECK-SAME:      start_index_map = [0]
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      slice_sizes = array<i64: 1, 1, 5>
// CHECK-SAME:    : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
func.func @index_select_to_gather_multi_offset_dims(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_larger_output
// CHECK:         "stablehlo.gather"(%arg0, %arg1)
// CHECK-SAME:      offset_dims = [3]
// CHECK-SAME:      collapsed_slice_dims = [0]
// CHECK-SAME:      start_index_map = [0]
// CHECK-SAME:      index_vector_dim = 3
// CHECK-SAME:      slice_sizes = array<i64: 1, 4>
// CHECK-SAME:    : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
func.func @index_select_to_gather_larger_output(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
  func.return %0 : tensor<1x3x1x4xf32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_regular_map
// CHECK:         "stablehlo.gather"(%arg0, %arg1)
// CHECK-SAME:      offset_dims = [1]
// CHECK-SAME:      collapsed_slice_dims = [0]
// CHECK-SAME:      start_index_map = [0]
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      slice_sizes = array<i64: 1, 4>
// CHECK-SAME:    : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<2x4xi32>
func.func @index_select_to_gather_regular_map(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi32>) -> tensor<2x4xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_reverse_map
// CHECK:         "stablehlo.gather"(%arg0, %arg1)
// CHECK-SAME:      offset_dims = [0]
// CHECK-SAME:      collapsed_slice_dims = [1]
// CHECK-SAME:      start_index_map = [1]
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      slice_sizes = array<i64: 3, 1>
// CHECK-SAME:    : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<3x2xi32>
func.func @index_select_to_gather_reverse_map(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi32>) -> tensor<3x2xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 1 : i64, batch_dims = 0 : i64}> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_batch_dim_greater_than_1
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<2x1xi32>
// CHECK:         %[[RESHAPE:.+]] = stablehlo.reshape %[[ARG1]] : (tensor<2xi32>) -> tensor<2x1xi32>
// CHECK:         %[[CONCAT:.+]] = stablehlo.concatenate %[[IOTA]], %[[RESHAPE]], dim = 1 : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>
// CHECK:         "stablehlo.gather"(%[[ARG0]], %[[CONCAT]])
// CHECK-SAME:      offset_dims = [1]
// CHECK-SAME:      collapsed_slice_dims = [0, 1]
// CHECK-SAME:      start_index_map = [0, 1]
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      slice_sizes = array<i64: 1, 1, 5>
// CHECK-SAME:    : (tensor<5x1x5xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
func.func @index_select_to_gather_batch_dim_greater_than_1(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 1 : i64, batch_dims = 1 : i64}> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_non_static_operand
// CHECK:         stablehlo.torch_index_select
func.func @index_select_to_gather_non_static_operand(%arg0 : tensor<5x1x?xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x1x?xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_non_static_index
// CHECK:         stablehlo.torch_index_select
func.func @index_select_to_gather_non_static_index(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<?xi32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x1x5xi32>, tensor<?xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_dim_less_than_batch_dims
// CHECK:         stablehlo.torch_index_select
func.func @index_select_to_gather_dim_less_than_batch_dims(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 1 : i64}> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

// CHECK-LABEL: func @index_select_to_gather_non_integer_index
// CHECK:         stablehlo.torch_index_select
func.func @index_select_to_gather_non_integer_index(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xf32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) <{dim = 0 : i64, batch_dims = 0 : i64}> : (tensor<5x1x5xi32>, tensor<2xf32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}
