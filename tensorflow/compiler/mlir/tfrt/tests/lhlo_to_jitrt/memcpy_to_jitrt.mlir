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

// CHECK: @gpu_memcpy_d2d(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>,
// CHECK:   %[[SRC:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_d2d(%dst: memref<?xf32>, %src: memref<?xf32>) {
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memcpy.d2d"}

// -----

// CHECK: @gpu_memcpy_h2d(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_h2d(%dst: memref<?xf32>, %dim: index) {
  // CHECK: %[[SRC:.*]] = memref.alloca
  %src = memref.alloca(%dim) : memref<?xf32>
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memcpy.h2d"}

// -----

// CHECK: @gpu_memcpy_d2h(
// CHECK:   %[[SRC:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_d2h(%src: memref<?xf32>, %dim: index) {
  // CHECK: %[[DST:.*]] = memref.alloca
  %dst = memref.alloca(%dim) : memref<?xf32>
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memcpy.d2h"}
