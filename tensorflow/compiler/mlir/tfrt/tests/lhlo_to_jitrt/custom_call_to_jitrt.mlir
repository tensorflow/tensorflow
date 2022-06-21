// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: lhlo-tfrt-opt %s -lmhlo-gpu-to-jitrt -split-input-file | FileCheck %s

// CHECK: func @test
// CHECK:   %[[ARG0:.*]]: memref<f32>
// CHECK: )
func.func @test(%arg0: memref<f32>) {
  // CHECK: call @[[CUSTOM_CALL:.*]](%[[ARG0]])
  // CHECK-SAME:   api_version = 2 : i32
  // CHECK-SAME:   backend_config = ""
  // CHECK-SAME:   call_target_name = "target"
  // CHECK-SAME: : (memref<f32>) -> ()
  "lmhlo.custom_call"(%arg0) {
    api_version = 2 : i32,
    backend_config = "",
    call_target_name = "target",
    operand_segment_sizes = dense<[0, 1]> : vector<2xi32>
  } : (memref<f32>) -> ()
  return
}

// CHECK: func.func private @[[CUSTOM_CALL]](memref<f32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.custom_call"}

// -----

// CHECK: func @test_with_mapping
// CHECK:   %[[ARG0:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG1:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG2:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG3:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG4:[0-9a-z]*]]: memref<f32>
// CHECK: )
func.func @test_with_mapping(
    %arg0: memref<f32>,
    %arg1: memref<f32>,
    %arg2: memref<f32>,
    %arg3: memref<f32>,
    %arg4: memref<f32>) {
  // CHECK: %[[HOLE:.*]] = memref.alloca() : memref<0xi8>

  // CHECK: call @[[CUSTOM_CALL:.*]](%[[ARG0]], %[[HOLE]], %[[ARG1]], %[[HOLE]],
  // CHECK-SAME:  %[[ARG2]], %[[ARG3]], %[[HOLE]], %[[ARG4]])
  // CHECK-SAME:   api_version = 1 : i32
  // CHECK-SAME:   backend_config = ""
  // CHECK-SAME:   call_target_name = "target"
  "lmhlo.custom_call"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    api_version = 1 : i32,
    backend_config = "",
    call_target_name = "target",
    operand_segment_sizes = dense<[2, 3]> : vector<2xi32>,
    target_arg_mapping = {
      args_to_target_args = [0, 2],
      num_args = 4 : i64,
      num_results = 4 : i64,
      results_to_target_results = [0, 1, 3]}
    } : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()

  return
}

// CHECK: func.func private @[[CUSTOM_CALL]](memref<f32>, memref<0xi8>,
// CHECK-SAME: memref<f32>, memref<0xi8>, memref<f32>, memref<f32>,
// CHECK-SAME: memref<0xi8>, memref<f32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.custom_call"}
