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
    %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Constant", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.LegacyCall"(%outputs) {_tpu_replicate = "cluster", device = "", f = @foo0} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  func.return
}
func.func @foo0(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xi32>
  }
  func.return %0 : tensor<*xi32>
}

// CHECK: node {
// CHECK:  name: "tf.LegacyCall"
// CHECK-NEXT:  op: "foo0"
// CHECK:  attr {
// CHECK-NEXT:  key: "_output_shapes"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         shape {
// CHECK:  attr {
// CHECK-NEXT:  key: "_tpu_replicate"
// CHECK-NEXT:    value {
// CHECK-NEXT:      s: "cluster"
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK: library {
// CHECK-NEXT:  function {
// CHECK-NEXT:    signature {
// CHECK-NEXT:      name: "foo0"
