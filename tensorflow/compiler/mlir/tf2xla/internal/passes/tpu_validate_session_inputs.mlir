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
// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-validate-session-inputs | FileCheck %s

// CHECK-LABEL: func @does_not_contian_InfeedDequeueTuple
func.func @does_not_contian_InfeedDequeueTuple(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----
func.func @contians_InfeedDequeueTuple(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-warning @+1 {{TPU_REPLICATED_CORE:0 device is not supported for op = tf.InfeedDequeueTuple in TF2XLA MLIR Bridge}}
    %infeed_output:3, %c2 = tf_executor.island wraps "tf.InfeedDequeueTuple"() {device = "/device:TPU_REPLICATED_CORE:0"} : () -> (tensor<3xi32>, tensor<4x?xf32>, tensor<*xi16>)
    %ro:2, %c3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----
func.func @graph_contains_v1_control_flow() {
  tf_executor.graph {
    // expected-warning @+1 {{ is v1 control flow op which is not supported in TF2XLA MLIR Bridge.}}
    %control = tf_executor.ControlTrigger {}
    tf_executor.fetch
  }
  func.return
}