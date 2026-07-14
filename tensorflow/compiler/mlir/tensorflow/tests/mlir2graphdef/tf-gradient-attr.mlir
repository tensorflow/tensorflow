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

func.func @main() {
  tf_executor.graph {
  // CHECK:      node {
  // CHECK-NEXT:   name: "Const"
  // CHECK-NEXT:   op: "Const"
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32> loc("Const")

  // CHECK:      node {
  // CHECK-NEXT:   name: "tf.PartitionedCall"
  // CHECK-NEXT:   op: "PartitionedCall"
  // CHECK:        func {
  // CHECK:          name: "foo"
  // CHECK:        }
    %1:2 = tf_executor.island wraps "tf.PartitionedCall"(%0) {Tin = [], Tout = [], config = "", config_proto = "", device = "", executor_type = "", f = @foo, name = "Call_foo"} : (tensor<f32>) -> tensor<*xf32>
    tf_executor.fetch
  }
  func.return
}

// CHECK:      library {
// CHECK-NEXT:   function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo"
// CHECK:      function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo_grad"
// CHECK:      gradient {
// CHECK-NEXT:     function_name: "foo"
// CHECK-NEXT:     gradient_func: "foo_grad"
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @foo_grad(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  func.return %graph : tensor<*xf32>
}

func.func @foo(%arg0: tensor<*xf32>) -> tensor<*xf32>
  attributes  {tf.gradient = @foo_grad} {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  func.return %graph : tensor<*xf32>
}
