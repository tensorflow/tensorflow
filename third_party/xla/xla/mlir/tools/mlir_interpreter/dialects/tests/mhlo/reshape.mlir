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

func.func @reshape() -> tensor<2x4x2xi32> {
  %cst = mhlo.constant dense<
    [[0, 1, 2, 3], [4, 5, 6, 7],
     [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %reshape = mhlo.reshape %cst : (tensor<4x4xi32>) -> tensor<2x4x2xi32>
  func.return %reshape : tensor<2x4x2xi32>
}

// CHECK-LABEL: @reshape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <2x4x2xi32>: [[[0, 1], [2, 3], [4, 5], [6, 7]],
// CHECK-SAME{LITERAL}:               [[8, 9], [10, 11], [12, 13], [14, 15]]]

func.func @reshape_0d_1d() -> tensor<1xi32> {
  %cst = mhlo.constant dense<42> : tensor<i32>
  %reshape = mhlo.reshape %cst : (tensor<i32>) -> tensor<1xi32>
  func.return %reshape : tensor<1xi32>
}

// CHECK-LABEL: @reshape_0d_1d
// CHECK-NEXT: Results
// CHECK-NEXT: [42]

func.func @reshape_1d_0d() -> tensor<i32> {
  %cst = mhlo.constant dense<42> : tensor<1xi32>
  %reshape = mhlo.reshape %cst : (tensor<1xi32>) -> tensor<i32>
  func.return %reshape : tensor<i32>
}

// CHECK-LABEL: @reshape_1d_0d
// CHECK-NEXT: Results
// CHECK-NEXT: 42
