// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: litert-opt %s --tfl-downcast-x64 --canonicalize | FileCheck %s

// CHECK-LABEL: testFuncSignature
// CHECK: (%arg0: tensor<4xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32>
func.func @testFuncSignature(%arg0: tensor<4xi64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: return %arg1 : tensor<2x2xf32>
  func.return %arg1 : tensor<2x2xf64>
}

// CHECK-LABEL: testConstantDowncast
func.func @testConstantDowncast() -> tensor<2xf64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<[1.0, 2.0]> : tensor<2xf64>
  func.return %0 : tensor<2xf64>
}

// CHECK-LABEL: testI64ConstantDowncast
func.func @testI64ConstantDowncast() -> tensor<i64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<42> : tensor<i32>
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<42> : tensor<i64>
  func.return %0 : tensor<i64>
}

// CHECK-LABEL: testGeneralOpForceConvert
func.func @testGeneralOpForceConvert(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK: %[[ADD:.*]] = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2xf32>
  // CHECK: return %[[ADD]]
  %0 = "tfl.add"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  func.return %0 : tensor<2xf64>
}