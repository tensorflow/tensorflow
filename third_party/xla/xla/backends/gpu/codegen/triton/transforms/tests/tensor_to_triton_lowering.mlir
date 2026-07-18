// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: xla-opt %s -split-input-file \
// RUN: -tensor-lower-to-triton \
// RUN: | FileCheck %s

//TODO(basioli): Consider fusing this and stablehlo_to_triton_lowering.mlir into xtile_to_triton_lowering.mlir

// CHECK: func @lower_bitcast(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<2x4x8xi32>
func.func @lower_bitcast(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xi32> {
  // CHECK: %[[RES:.*]] = tt.bitcast %[[ARG]] : tensor<2x4x8xf32> -> tensor<2x4x8xi32>
  %0 = tensor.bitcast %arg0 : tensor<2x4x8xf32> to tensor<2x4x8xi32>
  // CHECK: return %[[RES]] : tensor<2x4x8xi32>
  return %0 : tensor<2x4x8xi32>
}

// CHECK: func @lower_bitcast_0d(%[[ARG:.*]]: tensor<f32>) -> tensor<i32>
func.func @lower_bitcast_0d(%arg0: tensor<f32>) -> tensor<i32> {
  // CHECK: %[[SCALAR_ARG:.*]] = tensor.extract %[[ARG]][] : tensor<f32>
  // CHECK: %[[RES:.*]] = tt.bitcast %[[SCALAR_ARG]] : f32 -> i32
  // CHECK: %[[TENSOR_RES:.*]] = tensor.from_elements %[[RES]] : tensor<i32>
  %0 = tensor.bitcast %arg0 : tensor<f32> to tensor<i32>
  // CHECK: return %[[TENSOR_RES]] : tensor<i32>
  return %0 : tensor<i32>
}
