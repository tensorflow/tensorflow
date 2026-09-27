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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK: {tf_saved_model.index_path = ["input2"]}
// CHECK-SAME: {tf_saved_model.index_path = ["input1"]}
// CHECK-SAME: {tf_saved_model.index_path = ["start_logits"]}
// CHECK-SAME: {tf_saved_model.index_path = ["end_logits"]}
// CHECK-SAME: tf.entry_function = {inputs = "serving_default_input2:0,serving_default_input1:0", outputs = "StatefulPartitionedCall:1,StatefulPartitionedCall:0"}
// CHECK-SAME: tf_saved_model.exported_names = ["serving_default"]
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<?x384xf32> {tf_saved_model.index_path = ["input2"]}, %arg1: tensor<?x384xf32> {tf_saved_model.index_path = ["input1"]}) -> (tensor<?x5xf32> {tf_saved_model.index_path = ["start_logits"]}, tensor<?x5xf32> {tf_saved_model.index_path = ["end_logits"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input2:0,serving_default_input1:0", outputs = "StatefulPartitionedCall:1,StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<5xf32>
    %cst_0 = arith.constant dense<1.0> : tensor<5x384xf32>
    %cst_1 = arith.constant dense<1.0> : tensor<5x384xf32>
    %0 = "tfl.fully_connected"(%arg0, %cst_0, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x384xf32>, tensor<5x384xf32>, tensor<5xf32>) -> tensor<?x5xf32>
    %1 = "tfl.fully_connected"(%arg0, %cst_1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x384xf32>, tensor<5x384xf32>, tensor<5xf32>) -> tensor<?x5xf32>
    func.return %1, %0 : tensor<?x5xf32>, tensor<?x5xf32>
  }
}
