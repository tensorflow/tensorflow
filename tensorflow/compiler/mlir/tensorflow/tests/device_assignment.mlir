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
// RUN: tf-opt -tf-simple-device-assignment='default-device=gpu' %s | FileCheck %s

// CHECK-LABEL: func @device_test
func.func @device_test(%arg0: tensor<3x1xf32>) -> (tensor<3x3xf32>) {

  // CHECK: device = "gpu"
  %0 = "tf.Const"() {value = dense<[[1.0, 2.0, 3.0]]> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  // CHECK: device = "gpu"
  %1 = "tf.MatMul"(%arg0, %0) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  // CHECK: device = "cpu"
  %2 = "tf.Relu"(%1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK: device = "gpu"
  %3 = "tf.Relu"(%2) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"]} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  func.return %3 : tensor<3x3xf32>
}
