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

func.func @shape_cast() -> vector<2x2xi32> {
  %a = arith.constant dense<[[1, 2, 3, 4]]> : vector<1x4xi32>
  %cast = vector.shape_cast %a : vector<1x4xi32> to vector<2x2xi32>
  return %cast : vector<2x2xi32>
}

// CHECK-LABEL: @shape_cast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2], [3, 4]]

func.func @cast_of_transpose() -> (vector<3x2xi32>, vector<2x3xi32>) {
  %a = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : vector<2x3xi32>
  %b = vector.transpose %a, [1, 0] : vector<2x3xi32> to vector<3x2xi32>
  %cast = vector.shape_cast %b : vector<3x2xi32> to vector<2x3xi32>
  return %b, %cast : vector<3x2xi32>, vector<2x3xi32>
}

// CHECK-LABEL: @cast_of_transpose
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 4], [2, 5], [3, 6]]
// CHECK-NEXT{LITERAL}: [[1, 4, 2], [5, 3, 6]]
