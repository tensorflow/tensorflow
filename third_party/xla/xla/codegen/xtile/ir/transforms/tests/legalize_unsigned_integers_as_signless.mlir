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
// =============================================================================

// RUN: emitters_opt %s -legalize-unsigned-integers-as-signless -split-input-file -verify-diagnostics | FileCheck %s

func.func @unsigned_func(%arg0: ui32, %arg1: memref<10xui32>) -> ui32 {
  %c0 = arith.constant 0 : i32
  %c0_as_ui32 = builtin.unrealized_conversion_cast %c0 : i32 to ui32
  return %c0_as_ui32 : ui32
}
// CHECK: func.func @unsigned_func(%arg0: i32, %arg1: memref<10xi32>) -> i32 {
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[C0]] : i32 to i32
// CHECK:   return %[[CAST]] : i32
// CHECK: }

// -----

func.func @unsigned_dense_const() {
  %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xui16>
  return
}
// CHECK: func.func @unsigned_dense_const() {
// CHECK:   %[[C:.*]] = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi16>
// CHECK:   return
// CHECK: }
