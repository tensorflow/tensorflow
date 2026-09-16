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
// RUN: emitters_opt %s -split-input-file -xla-cpu-add-loop-unroll-flags | FileCheck %s

func.func @nested_for(%arg : tensor<16x16x8xf32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  scf.for %iter0 = %c0 to %c16 step %c1 iter_args(%res0 = %arg) -> tensor<16x16x8xf32> {
    scf.for %iter1 = %c0 to %c16 step %c1 iter_args(%res1 = %res0) -> tensor<16x16x8xf32> {
      scf.for %iter2 = %c0 to %c8 step %c1 iter_args(%res2 = %res1) -> tensor<16x16x8xf32> {
        %extracted = tensor.extract %res2[%iter0, %iter1, %iter2] : tensor<16x16x8xf32>
        scf.yield %res2 : tensor<16x16x8xf32>
      }
      scf.yield %res1 : tensor<16x16x8xf32>
    }
    scf.for %iter1 = %c0 to %c8 step %c1  iter_args(%res1 = %res0) -> tensor<16x16x8xf32> {
      %extracted = tensor.extract %res1[%iter0, %iter0, %iter1] : tensor<16x16x8xf32>
      scf.yield %res1 : tensor<16x16x8xf32>
    }
    scf.yield %res0 : tensor<16x16x8xf32>
  }
  return
}

// CHECK: #[[LOOP_UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK: #[[LOOP_ANNOTATION:.*]] = #llvm.loop_annotation<unroll = #[[LOOP_UNROLL]]>
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK: tensor.extract
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NOT: loop_annotation
// CHECK: scf.yield
// CHECK-NEXT: } {loop_annotation = #[[LOOP_ANNOTATION]]}
// CHECK-NEXT: scf.for
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: scf.yield
// CHECK-NEXT }
// CHECK-NOT: loop_annotation
// CHECK: scf.yield
// CHECK-NEXT: } {loop_annotation = #[[LOOP_ANNOTATION]]}
// CHECK-NEXT: return
