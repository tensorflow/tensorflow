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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Check that attributes that define derived shapes are exported.

// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape
// CHECK: unknown_rank: true
// CHECK: name: "static"
// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape {
// CHECK-NEXT: }
// CHECK: name: "static_10"
// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape
// CHECK: dim
// CHECK: size: 10

func.func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%1#0) {type = i32} : (tensor<i32>) -> tensor<*xi32> loc("unranked")
    %3:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%1#0) {type = i32} : (tensor<i32>) -> tensor<i32> loc("static")
    %4:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%0#0) {type = i32} : (tensor<10xi32>) -> tensor<10xi32> loc("static_10")
    tf_executor.fetch
  }
  func.return
}
