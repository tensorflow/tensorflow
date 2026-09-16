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
// RUN: tf-opt %s -tf-executor-tpu-v1-island-outlining | FileCheck %s

// CHECK: func @control_input
// CHECK-NOT: func @
// CHECK-LABEL: module @_tpu_v1_compat_outlined
// CHECK: @_tpu_v1_compat_outlined_func0
// CHECK: func @branch_0
// CHECK: func @branch_1
// CHECK: func @branch_2
// CHECK: func @branch_3
// CHECK: func @branch_4
module {
  func.func @control_input(%arg0: tensor<i1>) -> tensor<i32> {
    %0 = tf_executor.graph {
      %output, %control = tf_executor.island {
       "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
        %index = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i1>) -> tensor<i32>
        %input = "tf.opB"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i1>) -> tensor<i32>
        %result = "tf.Case"(%index, %input) {branches = [@branch_0, @branch_1, @branch_2, @branch_3, @branch_4], is_stateless = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        tf_executor.yield %result : tensor<i32>
      }
      tf_executor.fetch %output : tensor<i32>

    }
    func.return %0 : tensor<i32>
  }
  func.func @branch_0(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }
  func.func @branch_1(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }
  func.func @branch_2(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }
  func.func @branch_3(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }
  func.func @branch_4(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }
}
