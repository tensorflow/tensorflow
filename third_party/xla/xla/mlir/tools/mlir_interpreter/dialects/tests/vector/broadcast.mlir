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

func.func @scalar_to_unit() -> vector<i32> {
  %c1 = arith.constant 1 : i32
  %b = vector.broadcast %c1 : i32 to vector<i32>
  return %b : vector<i32>
}

// CHECK-LABEL: @scalar_to_unit
// CHECK-NEXT: Results
// CHECK-NEXT: vector<i32>: 1

func.func @scalar_to_2d() -> vector<2x3xi32> {
  %c1 = arith.constant 1 : i32
  %b = vector.broadcast %c1 : i32 to vector<2x3xi32>
  return %b : vector<2x3xi32>
}

// CHECK-LABEL: @scalar_to_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x3xi32>: [[1, 1, 1], [1, 1, 1]]

func.func @unit_to_unit() -> vector<i32> {
  %c1 = arith.constant dense<1> : vector<i32>
  %b = vector.broadcast %c1 : vector<i32> to vector<i32>
  return %b : vector<i32>
}

// CHECK-LABEL: @unit_to_unit
// CHECK-NEXT: Results
// CHECK-NEXT: vector<i32>: 1

func.func @stretch() -> vector<3xi32> {
  %c1 = arith.constant dense<1> : vector<1xi32>
  %b = vector.broadcast %c1 : vector<1xi32> to vector<3xi32>
  return %b : vector<3xi32>
}

// CHECK-LABEL: @stretch
// CHECK-NEXT: Results
// CHECK-NEXT: vector<3xi32>: [1, 1, 1]

func.func @stretch_and_broadcast() -> vector<2x1x2x2xi32> {
  %c1 = arith.constant dense<[[1], [2]]> : vector<2x1xi32>
  %b = vector.broadcast %c1 : vector<2x1xi32> to vector<2x1x2x2xi32>
  return %b : vector<2x1x2x2xi32>
}

// CHECK-LABEL: @stretch_and_broadcast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x1x2x2xi32>: [[[[1, 1], [2, 2]]], [[[1, 1], [2, 2]]]]
