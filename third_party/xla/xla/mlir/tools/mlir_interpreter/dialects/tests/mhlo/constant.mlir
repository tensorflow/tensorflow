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

func.func @main() -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>) {
  %i32 = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %f32 = mhlo.constant dense<[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]> : tensor<2x3xf32>
  %empty = mhlo.constant dense<> : tensor<0x0x3xi16>
  %scalar = mhlo.constant dense<3.14> : tensor<f64>
  return %i32, %f32, %empty, %scalar : tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>
}

// CHECK-LABEL: @main
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 1, 2], [3, 4, 5]]
// CHECK-NEXT{LITERAL}: [[0.000000e+00, 1.000000e-01, 2.000000e-01], [3.000000e-01, 4.000000e-01, 5.000000e-01]]
// CHECK-NEXT{LITERAL}: []
// CHECK-NEXT{LITERAL}: 3.140000e+00

func.func @ui8() -> tensor<ui8> {
  %v = mhlo.constant dense<123> : tensor<ui8>
  return %v : tensor<ui8>
}

// CHECK-LABEL: @ui8
// CHECK-NEXT: Results
// CHECK-NEXT: 123