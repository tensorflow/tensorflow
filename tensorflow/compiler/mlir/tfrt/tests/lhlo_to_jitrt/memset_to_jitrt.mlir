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

// RUN: lhlo-tfrt-opt %s -lmhlo-to-jitrt --split-input-file | FileCheck %s

// CHECK: @gpu_memset_i32(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xi32>
// CHECK: )
func.func @gpu_memset_i32(%dst: memref<?xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 0 : i32
  %cst = arith.constant 0 : i32
  // CHECK: call @[[MEMSET:.*]](%[[DST]], %[[CST]])
  gpu.memset %dst, %cst : memref<?xi32>, i32
  return
}

// CHECK: func private @[[MEMSET]](memref<?xi32>, i32)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memset"}

// -----

// CHECK: @gpu_memset_f32(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memset_f32(%dst: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: call @[[MEMSET:.*]](%[[DST]], %[[CST]])
  gpu.memset %dst, %cst : memref<?xf32>, f32
  return
}

// CHECK: func private @[[MEMSET]](memref<?xf32>, f32)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memset"}
