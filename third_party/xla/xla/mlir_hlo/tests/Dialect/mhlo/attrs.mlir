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
// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: parameter_replication
func.func @parameter_replication(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false, true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_empty(%arg0: tensor<f32> {mhlo.parameter_replication = []}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = []}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_single_false(%arg0: tensor<f32> {mhlo.parameter_replication = [false]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_single_true(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}


// -----

// expected-error@+1 {{parameter_replication: arg 0 has 1 leaf_buffers, but parameter_replication expects 2}}
func.func @parameter_replication_num_leaf_buffer_mismatch(%arg0: tensor<f32> {mhlo.parameter_replication = [true, false]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}