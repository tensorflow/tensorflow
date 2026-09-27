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

func.func @broadcast() -> (tensor<2x3xi32>, tensor<2x3xi32>) {
  %v = arith.constant dense<[1,2]> : tensor<2xi32>
  %init = tensor.empty() : tensor<2x3xi32>
  %bcast = linalg.broadcast
      ins(%v: tensor<2xi32>)
      outs(%init: tensor<2x3xi32>)
      dimensions = [1]
  func.return %init, %bcast : tensor<2x3xi32>, tensor<2x3xi32>
}

// CHECK-LABEL: @broadcast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0], [0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [2, 2, 2]]

func.func @bufferized() -> memref<2x3xi32> {
  %v = arith.constant dense<[1,2]> : memref<2xi32>
  %alloc = memref.alloc() : memref<2x3xi32>
  linalg.broadcast
      ins(%v: memref<2xi32>)
      outs(%alloc: memref<2x3xi32>)
      dimensions = [1]
  func.return %alloc : memref<2x3xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [2, 2, 2]]
