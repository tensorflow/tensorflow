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

func.func @select() -> (i32, i32) {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %r1 = arith.select %true, %c-1, %c1 : i32
  %r2 = arith.select %false, %c-1, %c1 : i32
  return %r1, %r2 : i32, i32
}

// CHECK-LABEL: @select
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1

func.func @vector() -> vector<4xi32> {
  %a = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  %b = arith.constant dense<[10, 20, 30, 40]> : vector<4xi32>
  %c = arith.constant dense<[true, false, true, false]> : vector<4xi1>
  %r = arith.select %c, %a, %b : vector<4xi1>, vector<4xi32>
  return %r : vector<4xi32>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<4xi32>: [1, 20, 3, 40]

func.func @scalar_vector() -> vector<4xi32> {
  %a = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  %b = arith.constant dense<[10, 20, 30, 40]> : vector<4xi32>
  %c = arith.constant false
  %r = arith.select %c, %a, %b : vector<4xi32>
  return %r : vector<4xi32>
}

// CHECK-LABEL: @scalar_vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<4xi32>: [10, 20, 30, 40]

func.func @tensor() -> tensor<4xi32> {
  %a = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %b = arith.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
  %c = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  %r = arith.select %c, %a, %b : tensor<4xi1>, tensor<4xi32>
  return %r : tensor<4xi32>
}

// CHECK-LABEL: @tensor
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<4xi32>: [1, 20, 3, 40]
