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

// RUN: lhlo-tfrt-opt %s --split-input-file \
// RUN:   -lmhlo-to-gpu-binary \
// RUN: | FileCheck %s

// CHECK: gpu.module @gpu_module
// CHECK-SAME: binary = "
// CHECK-SAME:   .visible .entry _pad_to_static(
// CHECK-SAME:     param .u64 _pad_to_static_param_0
// CHECK-SAME:     param .u64 _pad_to_static_param_1
// CHECK-SAME:     param .u64 _pad_to_static_param_2
// CHECK-SAME:   )
// CHECK-SAME: "} {

// CHECK: gpu.func @_pad_to_static(%[[ARG0:[a-z0-9]+]]: memref<4xf32>,
// CHECK-SAME:                     %[[ARG1:[a-z0-9]+]]: memref<4xf32>,
// CHECK-SAME:                     %[[ARG2:[a-z0-9]+]]: memref<i32>) kernel

// CHECK: @pad_to_static
func.func @pad_to_static(%arg0: memref<4xf32>,
                         %arg1: memref<4xf32>,
                         %arg2: memref<i32>) {
  "lmhlo.custom_call"(%arg0, %arg1, %arg2) {
    api_version = 1 : i32,
    backend_config = "",
    call_target_name = "PadToStatic",
    operand_segment_sizes = dense<[1, 2]> : vector<2xi32>
  } : (memref<4xf32>, memref<4xf32>, memref<i32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
