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

func.func @clamp() -> tensor<2x2xi32> {
  %lb = mhlo.constant dense<[[1, 7], [1, 7]]> : tensor<2x2xi32>
  %arg = mhlo.constant dense<[[4, 5], [6, 9]]> : tensor<2x2xi32>
  %ub = mhlo.constant dense<[[5, 9], [3, 6]]> : tensor<2x2xi32>
  %clamp = "mhlo.clamp"(%lb, %arg, %ub)
      : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %clamp : tensor<2x2xi32>
}

// CHECK-LABEL:         @clamp
// CHECK-NEXT:          Results
// CHECK-NEXT{LITERAL}: [[4, 7], [3, 6]]

func.func @clamp_f32() -> tensor<2x2xf32> {
  %lb = mhlo.constant dense<[[1.1, 7.1], [1.1, 7.1]]> : tensor<2x2xf32>
  %arg = mhlo.constant dense<[[4.1, 5.1], [6.1, 9.1]]> : tensor<2x2xf32>
  %ub = mhlo.constant dense<[[5.1, 9.1], [3.1, 6.1]]> : tensor<2x2xf32>
  %clamp = "mhlo.clamp"(%lb, %arg, %ub)
      : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %clamp : tensor<2x2xf32>
}

// CHECK-LABEL:         @clamp
// CHECK-NEXT:          Results
// CHECK-NEXT{LITERAL}: [[4.100000e+00, 7.100000e+00], [3.100000e+00, 6.100000e+00]]
