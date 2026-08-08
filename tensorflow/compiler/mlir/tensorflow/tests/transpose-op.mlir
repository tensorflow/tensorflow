// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s -split-input-file -verify-diagnostics

func.func @out_of_bounds_check(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[0, 0x4141, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  // expected-error @+1 {{'tf.Transpose' op perm[1]=16705 must be in range [-4, 4)}}
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x8xf32>
  func.return %3 : tensor<1x4x4x8xf32>
}