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

// RUN: lhlo-tfrt-opt %s -gpu-to-jitrt | FileCheck %s

// Check that all gpu dialect `launch_func` operations gets converted to
// function calls bound to jitrt custom calls.

module attributes {gpu.container_module} {

// CHECK-NOT: gpu.module
gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @main(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(
// CHECK:   %[[ARG0:.*]]: memref<4x4xf32>,
// CHECK:   %[[ARG1:.*]]: memref<4x4xf32>
// CHECK: )
func.func @func(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
  // Launch dimensions converted to i32 as a part of the lowering.
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK: %[[C6:.*]] = arith.constant 6 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // CHECK: call @[[LAUNCH:[_a-z.]+]](%[[C1]], %[[C2]], %[[C3]], %[[C4]],
  // CHECK-SAME: %[[C5]], %[[C6]], %[[ARG0]], %[[ARG1]])
  // CHECK-DAG: kernel = "main"
  gpu.launch_func  @gpu_module::@main
    blocks in (%c1, %c2, %c3)
    threads in (%c4, %c5, %c6)
    args(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>)

  func.return
}

// CHECK: func private @[[LAUNCH]](i32, i32, i32, i32, i32, i32,
// CHECK-SAME: memref<4x4xf32>, memref<4x4xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.func.launch"}

}
