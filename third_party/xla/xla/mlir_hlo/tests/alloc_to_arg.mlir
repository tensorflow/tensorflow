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
// RUN: mlir-hlo-opt %s --alloc-to-arg -verify-diagnostics -split-input-file -allow-unregistered-dialect \
// RUN: | FileCheck %s

// CHECK-LABEL: func @alloc_to_arg
// CHECK-SAME: (%arg0: memref<8xf32>, %arg1: memref<8xf32> {my.attr})
func.func @alloc_to_arg(%arg0: memref<8xf32>) -> (memref<8xf32> {my.attr}) {
  // CHECK-NOT: memref.alloc
  %0 = memref.alloc() : memref<8xf32>
  return %0 : memref<8xf32>
}

// -----

func.func @not_alloc(%arg0: memref<8xf32>) -> memref<8xf32> {
  // expected-error@+1 {{expected operand #0 to be defined by (shape-expanded) memref.alloc}}
  return %arg0 : memref<8xf32>
}

// -----

// CHECK: @fusion(%[[ARG0:.*]]: memref<4x4x8x32xf32>)
func.func @fusion() -> memref<4x4x8x32xf32> {
  // CHECK:   %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3{{\]\]}}
  // CHECK:   "some.use"(%[[COLLAPSE_SHAPE]], %[[ARG0]])
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x32xf32>
  %expand_shape = memref.expand_shape %alloc [[0, 1, 2], [3]] output_shape [4, 4, 8, 32] : memref<128x32xf32> into memref<4x4x8x32xf32>
  "some.use"(%alloc, %expand_shape) : (memref<128x32xf32>, memref<4x4x8x32xf32>) -> ()
  return %expand_shape : memref<4x4x8x32xf32>
}
