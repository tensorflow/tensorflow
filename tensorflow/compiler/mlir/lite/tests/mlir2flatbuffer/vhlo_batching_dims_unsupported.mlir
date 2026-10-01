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
// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s 2>&1 | FileCheck %s

// CHECK: error: stablehlo.gather with batching dims is not supported
// CHECK: error: stablehlo.scatter with batching dims is not supported

func.func @main(%operand: tensor<2x3x4x2xi32>, %indices: tensor<2x2x3x2xi64>,
                %updates: tensor<2x2x3x2x2xi32>) -> (tensor<2x2x3x2x2xi32>, tensor<2x3x4x2xi32>) {
  %0 = "vhlo.gather_v2"(%operand, %indices) <{
    offset_dims = #vhlo.tensor_v1<dense<[3, 4]> : tensor<2xi64>>,
    collapsed_slice_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>,
    operand_batching_dims = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    start_indices_batching_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>,
    start_index_map = #vhlo.tensor_v1<dense<[2, 1]> : tensor<2xi64>>,
    index_vector_dim = #vhlo.integer_v1<3 : i64>,
    slice_sizes = #vhlo.tensor_v1<dense<[1, 1, 2, 2]> : tensor<4xi64>>,
    indices_are_sorted = #vhlo.bool_v1<false>
  }> : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
  %1 = "vhlo.scatter_v2"(%operand, %indices, %updates) <{
    update_window_dims = #vhlo.tensor_v1<dense<[3, 4]> : tensor<2xi64>>,
    inserted_window_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>,
    input_batching_dims = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    scatter_indices_batching_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>,
    scatter_dims_to_operand_dims = #vhlo.tensor_v1<dense<[2, 1]> : tensor<2xi64>>,
    index_vector_dim = #vhlo.integer_v1<3 : i64>,
    indices_are_sorted = #vhlo.bool_v1<false>,
    unique_indices = #vhlo.bool_v1<false>}> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    "vhlo.return_v1"(%rhs) : (tensor<i32>) -> ()
  }) : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi32>) -> tensor<2x3x4x2xi32>
  func.return %0, %1 : tensor<2x2x3x2x2xi32>, tensor<2x3x4x2xi32>
}
