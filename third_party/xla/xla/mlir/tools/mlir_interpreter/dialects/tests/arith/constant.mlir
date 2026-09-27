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

func.func @tensor() -> tensor<2xi16> {
  %cst = arith.constant dense<[42, 43]> : tensor<2xi16>
  return %cst : tensor<2xi16>
}

// CHECK-LABEL: @tensor
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 43]

func.func @tensor_splat() -> tensor<2xi32> {
  %cst = arith.constant dense<42> : tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: @tensor_splat
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 42]

func.func @scalar() -> i1 {
  %cst = arith.constant true
  return %cst : i1
}

// CHECK-LABEL: @scalar
// CHECK-NEXT: Results
// CHECK-NEXT: true

func.func @vector() -> vector<2x3xi32> {
  %cst = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : vector<2x3xi32>
  return %cst : vector<2x3xi32>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x3xi32>: [[1, 2, 3], [4, 5, 6]]
