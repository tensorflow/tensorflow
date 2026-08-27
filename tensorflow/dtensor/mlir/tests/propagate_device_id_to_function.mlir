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
// RUN: dtensor-opt %s -split-input-file -dtensor-propagate-device-id-to-function-args | FileCheck %s

// CHECK-LABEL: func @main
// CHECK-SAME:  %[[ARG_0:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[ARG_1:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[ARG_2:[a-z0-9]+]]: tensor<i32>
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<i32>{ tf._layout = ["\0A\00\12\C6\01\0A\05\0A\01x\10\02\0A\05\0A\01y\10\02\12,/job:localhost/replica:0/task:0/device:CPU:0\12,/job:localhost/replica:0/task:0/device:CPU:1\12,/job:localhost/replica:0/task:0/device:CPU:2\12,/job:localhost/replica:0/task:0/device:CPU:3"]},
  %arg2: tensor<i32>{ tf._layout = ["\0A\00\12\C6\01\0A\05\0A\01x\10\02\0A\05\0A\01y\10\02\12,/job:localhost/replica:0/task:0/device:CPU:0\12,/job:localhost/replica:0/task:0/device:CPU:1\12,/job:localhost/replica:0/task:0/device:CPU:2\12,/job:localhost/replica:0/task:0/device:CPU:3"]}) -> (tensor<i32>) {
  // CHECK: "tf.StatefulPartitionedCall"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]])
  %1 = "tf.StatefulPartitionedCall"(%arg1, %arg2) {f = @callee1, config = "", config_proto = "", executor_type = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func private @callee1
// CHECK-SAME:  %[[CALL1_ARG0:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[CALL1_ARG1:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[CALL1_ARG2:[a-z0-9]+]]: tensor<i32>
func.func private @callee1(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {tf.signature.is_stateful} {
  // CHECK:      "tf.PartitionedCall"(%[[CALL1_ARG0]], %[[CALL1_ARG1]], %[[CALL1_ARG2]])
  // CHECK-SAME: {config = "", config_proto = "", executor_type = "", f = @callee2}
  // CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0, %arg1) {f = @callee2, config = "", config_proto = "", executor_type = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func private @callee2
// CHECK-SAME:  %[[CALL2_ARG0:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[CALL2_ARG1:[a-z0-9]+]]: tensor<i32>
// CHECK-SAME:  %[[CALL2_ARG2:[a-z0-9]+]]: tensor<i32>
func.func private @callee2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {tf.signature.is_stateful} {
  // CHECK: "tf.Add"(%[[CALL2_ARG1]], %[[CALL2_ARG2]])
  %1 = "tf.Add"(%arg0, %arg1) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}
