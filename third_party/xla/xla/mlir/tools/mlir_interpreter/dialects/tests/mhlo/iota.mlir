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

func.func @iota_f32() -> tensor<1x2x3x4xf32> {
  %result = "mhlo.iota"() {
    iota_dimension = 2 : i64
  } : () -> tensor<1x2x3x4xf32>
  func.return %result : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: @iota_f32
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
// CHECK{LITERAL}:         [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
// CHECK{LITERAL}:         [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]]]

func.func @iota_i32() -> tensor<1x2x3x4xi32> {
  %result = "mhlo.iota"() {
    iota_dimension = 3 : i64
  } : () -> tensor<1x2x3x4xi32>
  func.return %result : tensor<1x2x3x4xi32>
}

// CHECK-LABEL: @iota_i32
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[[0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3]],
// CHECK{LITERAL}         [[0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3],
// CHECK{LITERAL}          [0, 1, 2, 3]]]]
