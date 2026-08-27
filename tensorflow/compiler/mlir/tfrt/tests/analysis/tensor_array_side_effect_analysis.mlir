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
// RUN: tf-tfrt-opt -tfrt-test-tensor-array-effect -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @test_tensor_array_effect
// expected-remark@+1 {{HasAtMostTensorArrayEffect: 1}}
func.func @test_tensor_array_effect(%index: tensor<i32>, %size: tensor<i32>, %flow_0: tensor<f32>, %flow_1: tensor<f32>, %handle_0: tensor<2x!tf_type.resource<tensor<?x100xf32>>>, %handle_1: tensor<2x!tf_type.resource<tensor<?x512xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>) {
  %cst = "tf.Const"() {value = dense<1.1> : tensor<100x512xf32>} : () -> tensor<100x512xf32>
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %x = "tf.TensorArrayReadV3"(%handle_0, %index, %flow_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<i32>, tensor<f32>) -> tensor<?x100xf32>
  %y = "tf.MatMul"(%x, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x100xf32>, tensor<100x512xf32>) -> (tensor<?x512xf32>)
  %flow_1_out = "tf.TensorArrayWriteV3"(%handle_1, %index, %y, %flow_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<i32>, tensor<?x512xf32>, tensor<f32>) -> tensor<f32>
  %next_index = "tf.AddV2"(%index, %one) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %next_index, %size, %flow_0, %flow_1_out, %handle_0, %handle_1 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>
}
