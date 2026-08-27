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

func.func @convert_i1_to_f32() -> tensor<2xf32> {
  %input = mhlo.constant dense<[true, false]> : tensor<2xi1>
  %result = "mhlo.convert"(%input) : (tensor<2xi1>) -> tensor<2xf32>
  func.return %result : tensor<2xf32>
}

// CHECK-LABEL: @convert_i1_to_f32
// CHECK-NEXT: Results
// CHECK-NEXT: [1.000000e+00, 0.000000e+00]

func.func @convert_f32_to_i16() -> tensor<2xi16> {
  %input = mhlo.constant dense<[1.4, 2.55]> : tensor<2xf32>
  %result = "mhlo.convert"(%input) : (tensor<2xf32>) -> tensor<2xi16>
  func.return %result : tensor<2xi16>
}

// CHECK-LABEL: @convert_f32_to_i16
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 2]
