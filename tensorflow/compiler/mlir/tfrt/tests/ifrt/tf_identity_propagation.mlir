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
// RUN: tf-tfrt-opt %s -tf-identity-propagation -canonicalize | FileCheck %s

// CHECK-LABEL: func @identity
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>)
func.func @identity(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT: "tf.Identity"
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ARG0]]
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @identity_terminator
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>)
func.func @identity_terminator(%arg0: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
  // CHECK: %[[IDENTITY:.*]] = "tf.Identity"
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<*xi32>
  // CHECK-NOT: "tf.Identity"
  %1 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[IDENTITY]], %[[ARG0]]
  func.return %0, %1 : tensor<*xi32>, tensor<i32>
}

// CHECK-LABEL: func @xla_sharding
func.func @xla_sharding(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: %[[OUTPUT:.*]] = "tf.Identity"
  %0 = "tf.Identity"(%arg0) {_XlaSharding = ""} : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[OUTPUT]]
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @identity_n
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<f32>)
func.func @identity_n(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  // CHECK-NOT: "tf.IdentityN"
  %0:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  // CHECK: return %[[ARG0]], %[[ARG1]]
  func.return %0#0, %0#1 : tensor<i32>, tensor<f32>
}
