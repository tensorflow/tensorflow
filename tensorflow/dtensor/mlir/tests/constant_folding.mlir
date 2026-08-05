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
// RUN: dtensor-opt %s -split-input-file -dtensor-constant-folding | FileCheck %s

// Check that constants with same size/value are de-duplicated.
// CHECK-LABEL: func @check_constants_folded
func.func @check_constants_folded() {
  // CHECK:      %[[CONST_OUT_0:.*]] = "tf.Const"()
  // CHECK-SAME: value = dense<[8, 128, 128]> : tensor<3xi32>
  // CHECK-NEXT: %[[CONST_OUT_1:.*]] = "tf.Const"()
  // CHECK-SAME: value = dense<[8, 128]> : tensor<2xi32>
  // CHECK-NEXT: "tf.A"(%[[CONST_OUT_0]], %[[CONST_OUT_0]], %[[CONST_OUT_0]], %[[CONST_OUT_1]], %[[CONST_OUT_1]])
  // CHECK-NEXT: return
  %1 = "tf.Const"() {value = dense<[8, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tf.Const"() {value = dense<[8, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tf.Const"() {value = dense<[8, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %4 = "tf.Const"() {value = dense<[8, 128]> : tensor<2xi32>} : () -> tensor<2xi32>
  %5 = "tf.Const"() {value = dense<[8, 128]> : tensor<2xi32>} : () -> tensor<2xi32>
  "tf.A"(%1, %2, %3, %4, %5) : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<2xi32>, tensor<2xi32>) -> ()
  func.return
}
