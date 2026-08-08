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

func.func @transpose() -> vector<2x1x4x3xi32> {
  %0 = arith.constant dense<[[[
      [000, 001, 002, 003],
      [010, 011, 012, 013],
      [020, 021, 022, 023]
    ],
    [
      [100, 101, 102, 103],
      [110, 111, 112, 113],
      [120, 121, 122, 123]
    ]]]> : vector<1x2x3x4xi32>
  %1 = vector.transpose %0, [1, 0, 3, 2]
    : vector<1x2x3x4xi32> to vector<2x1x4x3xi32>
  return %1 : vector<2x1x4x3xi32>
}

// CHECK-LABEL: @transpose
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[[0, 10, 20],
// CHECK{LITERAL}:         [1, 11, 21],
// CHECK{LITERAL}:         [2, 12, 22],
// CHECK{LITERAL}:         [3, 13, 23]]],
// CHECK{LITERAL}:       [[[100, 110, 120],
// CHECK{LITERAL}:         [101, 111, 121],
// CHECK{LITERAL}:         [102, 112, 122],
// CHECK{LITERAL}:         [103, 113, 123]]]]
