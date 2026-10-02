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

func.func @fill() -> (tensor<2xi32>, tensor<2xi32>) {
  %c42 = arith.constant 42 : i32
  %init = tensor.empty() : tensor<2xi32>
  %fill = linalg.fill ins(%c42 : i32) outs(%init : tensor<2xi32>) -> tensor<2xi32>
  func.return %init, %fill : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: @fill
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 0]
// CHECK-NEXT{LITERAL}: [42, 42]

func.func @bufferized() -> memref<2xi32> {
  %c42 = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<2xi32>
  linalg.fill ins(%c42 : i32) outs(%alloc : memref<2xi32>)
  func.return %alloc : memref<2xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [42, 42]
