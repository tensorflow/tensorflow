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

func.func @insert() -> tensor<1x3x3xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]> : tensor<1x3x3xi32>
  %cst_2 = arith.constant dense<[[[10], [11]]]> : tensor<1x2x1xi32>
  %ret = tensor.insert_slice %cst_2 into %cst[0, 1, 1][1, 2, 1][1, 1, 1]
    : tensor<1x2x1xi32> into tensor<1x3x3xi32>
  return %ret : tensor<1x3x3xi32>
}

// CHECK-LABEL: @insert
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 10, 6], [7, 11, 9]]]

func.func @rank_increase() -> tensor<1x3x3xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]> : tensor<1x3x3xi32>
  %cst_2 = arith.constant dense<[10, 11]> : tensor<2xi32>
  %ret = tensor.insert_slice %cst_2 into %cst[0, 1, 1][1, 2, 1][1, 1, 1]
    : tensor<2xi32> into tensor<1x3x3xi32>
  return %ret : tensor<1x3x3xi32>
}

// CHECK-LABEL: @rank_increase
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 10, 6], [7, 11, 9]]]
