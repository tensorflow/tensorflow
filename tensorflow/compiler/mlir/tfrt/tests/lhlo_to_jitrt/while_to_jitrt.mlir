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

// RUN: lhlo-tfrt-opt %s -lmhlo-gpu-to-jitrt | FileCheck %s

module attributes {gpu.container_module} {
  memref.global "private" constant @constant : memref<i32> = dense<0>

  gpu.module @cond attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>, %arg1: memref<i1>) kernel {
      gpu.return
    }
  }

  gpu.module @body attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>) kernel {
      gpu.return
    }
  }

  // CHECK:      @while_loop(
  // CHECK-SAME:   %[[ARG0:.*]]: memref<i32>,
  // CHECK-SAME:   %[[ARG1:.*]]: memref<i1>
  // CHECK-SAME: )
  func.func @while_loop(%arg0: memref<i32>, %arg1: memref<i1>) {
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @constant : memref<i32>
    gpu.memcpy  %arg0, %0 : memref<i32>, memref<i32>

    // CHECK: scf.while : () -> ()
    "lmhlo.while"(%arg1) ({
      // CHECK: gpu.launch_func @cond::@fn
      // CHECK: %[[HOST_PRED:.*]] = memref.alloca() : memref<i1>
      // CHECK: gpu.memcpy %[[HOST_PRED]], %[[ARG1]]
      // CHECK: %[[COND:.*]] = memref.load %[[HOST_PRED]][] : memref<i1>
      // CHECK: scf.condition(%[[COND]])
      gpu.launch_func @cond::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>, %arg1 : memref<i1>)
      "lmhlo.terminator"() : () -> ()
    }, {
      // CHECK: gpu.launch_func @body::@fn
      // CHECK: scf.yield
      gpu.launch_func @body::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }) : (memref<i1>) -> ()
    "lmhlo.terminator"() : () -> ()
  }
}
