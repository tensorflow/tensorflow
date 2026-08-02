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
// RUN: fusion_compiler_opt %s \
// RUN: -xtile-cpu-memref-copy-to-loops -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: @identity_copy_is_unchanged
func.func @identity_copy_is_unchanged(%arg0: memref<5xi32>, %arg1: memref<5xi32>) {
  // CHECK: memref.copy
  memref.copy %arg0, %arg1 : memref<5xi32> to memref<5xi32>
  func.return
}


// CHECK-LABEL: @non_default_layout_copy_to_loops
func.func @non_default_layout_copy_to_loops(
    %arg0: memref<5x2xf32, strided<[1, 5]>>,
    %arg1: memref<5x2xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: scf.for %[[IDX0:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:   scf.for %[[IDX1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
  // CHECK:     %[[ELEMENT:.*]] = memref.load %arg0[%[[IDX0]], %[[IDX1]]]
  // CHECK-SAME: : memref<5x2xf32, strided<[1, 5]>>
  // CHECK:     memref.store %[[ELEMENT]], %arg1[%[[IDX0]], %[[IDX1]]]
  // CHECK-SAME: : memref<5x2xf32>
  // CHECK:   }
  // CHECK: }
  memref.copy %arg0, %arg1 : memref<5x2xf32, strided<[1, 5]>> to memref<5x2xf32>
  func.return
}
