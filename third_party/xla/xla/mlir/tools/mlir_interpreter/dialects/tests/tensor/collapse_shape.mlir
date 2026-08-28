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

func.func @collapse_shape()
    -> (tensor<1x6xi32>, tensor<2x3xi32>, tensor<6xi32>) {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %collapse1 = tensor.collapse_shape %cst [[0], [1, 2]]
      : tensor<1x2x3xi32> into tensor<1x6xi32>
  %collapse2 = tensor.collapse_shape %cst [[0, 1], [2]]
      : tensor<1x2x3xi32> into tensor<2x3xi32>
  %collapse3 = tensor.collapse_shape %cst [[0, 1, 2]]
      : tensor<1x2x3xi32> into tensor<6xi32>
  return %collapse1, %collapse2, %collapse3
      : tensor<1x6xi32>, tensor<2x3xi32>, tensor<6xi32>
}

// CHECK-LABEL: @collapse_shape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4, 5, 6]]
// CHECK-NEXT{LITERAL}: [[1, 2, 3], [4, 5, 6]]
// CHECK-NEXT{LITERAL}: [1, 2, 3, 4, 5, 6]

func.func @to_unit() -> tensor<i32> {
  %cst = arith.constant dense<42> : tensor<1x1x1x1xi32>
  %collapse = tensor.collapse_shape %cst []
    : tensor<1x1x1x1xi32> into tensor<i32>
  return %collapse : tensor<i32>
}

// CHECK-LABEL: @to_unit
// CHECK-NEXT: Results
// CHECK-NEXT: 42

func.func @dynamic() -> tensor<?xi32> {
  %cst = arith.constant dense<42> : tensor<2x3xi32>
  %cast = tensor.cast %cst : tensor<2x3xi32> to tensor<?x3xi32>
  %collapse = tensor.collapse_shape %cast [[0, 1]] : tensor<?x3xi32> into tensor<?xi32>
  return %collapse : tensor<?xi32>
}

// CHECK-LABEL: @dynamic
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 42, 42, 42, 42, 42]
