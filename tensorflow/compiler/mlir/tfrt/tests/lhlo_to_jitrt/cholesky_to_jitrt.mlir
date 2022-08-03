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

// CHECK: @compute(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG1:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG2:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG3:[a-z0-9]+]]: memref<4x4xi32>
// CHECK: )
func.func @compute(%operand: memref<4x4xi32>, %a: memref<4x4xi32>,
                   %workspace: memref<4x4xi32>, %info: memref<4x4xi32>) {

  // CHECK: call @[[CHOLESKY:.*]](%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
  // CHECK-SAME: batch_size = 1 : i64
  // CHECK-SAME: is_lower = true
  // CHECK-SAME: n = 4 : i64
  "lmhlo_gpu.cholesky"(%operand, %a, %workspace, %info) {
    batch_size = 1 : i64,
    is_lower = true,
    n = 4 : i64
  } : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> ()

  // CHECK-NEXT: return
  func.return
}

// CHECK: func private @[[CHOLESKY]](memref<4x4xi32>, memref<4x4xi32>,
// CHECK-SAME:                       memref<4x4xi32>, memref<4x4xi32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.cholesky"}
