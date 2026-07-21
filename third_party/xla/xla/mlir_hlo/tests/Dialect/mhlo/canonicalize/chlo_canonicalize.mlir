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
// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: constant_like_constant
func.func @constant_like_constant(%arg0: tensor<3x4xi32>) -> tensor<3x4xf32> {
  // CHECK: chlo.constant dense<3.200000e+00>
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<3x4xi32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: constant_like_constant_dynamic
func.func @constant_like_constant_dynamic(%arg0: tensor<?x?xi32>) -> tensor<?x?xf32> {
  // CHECK: chlo.constant_like
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
