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

func.func @dot_2d() -> tensor<2x2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %dot : tensor<2x2xi32>
}

// CHECK-LABEL: @dot_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[16, 19], [36, 43]]

func.func @dot_2d_1d() -> tensor<2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %dot : tensor<2xi32>
}

// CHECK-LABEL: @dot_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [14, 32]

func.func @dot_1d_1d() -> tensor<i32> {
  %lhs = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<i32>
  return %dot : tensor<i32>
}

// CHECK-LABEL: @dot_1d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: 14