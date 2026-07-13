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
// RUN: tf-tfrt-opt -optimize-tf-for-tfrt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @fold_device_index
func.func @fold_device_index() -> tensor<i32> {
  // CHECK-NOT: tf.DeviceIndex
  // CHECK: tf.Const
  // CHECK-SAME: value = dense<1> : tensor<i32>
  %0 = "tf.DeviceIndex"() {device = "/device:CPU:0", device_names = ["GPU", "CPU"]} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @not_fold_device_index
func.func @not_fold_device_index() -> tensor<i32> {
  // CHECK-NOT: tf.Const
  // CHECK: tf.DeviceIndex
  %0 = "tf.DeviceIndex"() {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @eliminate_multinomial
func.func @eliminate_multinomial(%0: tensor<*xf32>, %1: tensor<*xi32>) -> (tensor<*xi64>, tensor<*xi64>) {
  // CHECK-NEXT: tf.Multinomial
  // CHECK-NEXT: return
  %2 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  %3 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  func.return %2, %3 : tensor<*xi64>, tensor<*xi64>
}

// -----

// CHECK-LABEL: @not_eliminate_multinomial
func.func @not_eliminate_multinomial(%0: tensor<*xf32>, %1: tensor<*xi32>) -> (tensor<*xi64>, tensor<*xi64>) {
  // CHECK-NEXT: tf.Multinomial
  // CHECK-SAME: seed = 0
  // CHECK-NEXT: tf.Multinomial
  // CHECK-SAME: seed = 1
  // CHECK-NEXT: tf.Multinomial
  // CHECK-SAME: seed = 0
  // CHECK-NEXT: return
  %2 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  %3 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 1 : i64, seed2 = 1 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  %4 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  %5 = "tf.Multinomial"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xi64>
  func.return %2, %3 : tensor<*xi64>, tensor<*xi64>
}
