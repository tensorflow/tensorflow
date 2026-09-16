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
// RUN: tf-opt %s -split-input-file -tf-device-convert-launch-func-to-tf-call | FileCheck %s

// Tests a single `tf_device.launch_func`.

// CHECK-LABEL: func @single_launch_func
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xf32>)
func.func @single_launch_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT:[0-9]*]] = "tf.PartitionedCall"(%[[A_OUTPUT]])
      // CHECK-SAME: f = @_func
      // CHECK-SAME: device = "/device:test_device:0"
      %3 = "tf_device.launch_func"(%2) {device = "/device:test_device:0", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: tf_executor.yield %[[CALL_OUTPUT]]
      tf_executor.yield %3 : tensor<?xf32>
    }
    tf_executor.fetch %1#0 : tensor<?xf32>
  }
  func.return %0 : tensor<?xf32>
}

func.func @_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  func.return %arg0 : tensor<?xf32>
}

// -----

// Tests multiple `tf_device.launch_func`.

// CHECK-LABEL: func @multi_launch_func
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xf32>)
func.func @multi_launch_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT_0:[0-9]*]] = "tf.PartitionedCall"(%[[A_OUTPUT]])
      // CHECK-SAME: f = @_func
      // CHECK-SAME: device = "/device:test_device:0"
      %3 = "tf_device.launch_func"(%2) {device = "/device:test_device:0", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT_1:[0-9]*]] = "tf.PartitionedCall"(%[[CALL_OUTPUT_0]])
      // CHECK-SAME: f = @_func
      // CHECK-SAME: device = "/device:test_device:1"
      %4 = "tf_device.launch_func"(%3) {device = "/device:test_device:1", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: tf_executor.yield %[[CALL_OUTPUT_1]]
      tf_executor.yield %4 : tensor<?xf32>
    }
    tf_executor.fetch %1#0 : tensor<?xf32>
  }
  func.return %0 : tensor<?xf32>
}

func.func @_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  func.return %arg0 : tensor<?xf32>
}
