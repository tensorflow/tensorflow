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
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @clone() -> (memref<i32>, memref<i32>) {
  %a = arith.constant dense<1> : memref<i32>
  %b = bufferization.clone %a : memref<i32> to memref<i32>
  %c = arith.constant 2 : i32
  memref.store %c, %b[] : memref<i32>
  return %a, %b : memref<i32>, memref<i32>
}

// CHECK-LABEL: @clone
// CHECK-NEXT: Results
// CHECK-NEXT: TensorOrMemref<i32>: 1
// CHECK-NEXT: TensorOrMemref<i32>: 2
