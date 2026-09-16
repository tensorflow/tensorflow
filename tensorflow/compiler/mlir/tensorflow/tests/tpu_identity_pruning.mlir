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
// RUN: tf-opt %s -tf-tpu-identity-pruning | FileCheck %s --dump-input=always

// Tests Identity op in cluster is pruned away.

// CHECK-LABEL: func @testIdentity
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @testIdentity(%arg0: tensor<i32>) {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[ARG0]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) : () -> tensor<i32>
  func.return
}

// Tests IdentityN op in cluster is pruned away.

// CHECK-LABEL: func @testIdentityN
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>, [[ARG1:%.*]]: tensor<f32>)
func.func @testIdentityN(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  // CHECK-NOT:  "tf.IdentityN"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[ARG0]], [[ARG1]]
  %0:2 = "tf_device.cluster"() ({
    %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    tf_device.return %1#0, %1#1 : tensor<i32>, tensor<f32>
  }) : () -> (tensor<i32>, tensor<f32>)
  func.return
}

// Tests transitive Identity ops reachable from the cluster are pruned away.

// CHECK-LABEL: func @testTransitiveIdentity
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @testTransitiveIdentity(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      "tf.PartitionedCall"([[ARG0]])
  // CHECK-SAME: f = @callee0
  %0 = "tf_device.cluster"() ({
    %1 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @callee0} : (tensor<i32>) -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) : () -> tensor<i32>
  func.return
}

// CHECK-LABEL: func @callee0
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @callee0(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      "tf.PartitionedCall"([[ARG0]])
  // CHECK-SAME: f = @callee1
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @callee1} : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @callee1
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @callee1(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      return [[ARG0]]
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// Tests Identity ops not reachable from the cluster are not pruned away.

// CHECK-LABEL: func @testIdentityOutsideCluster
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @testIdentityOutsideCluster(%arg0: tensor<i32>) {
  // CHECK:      [[IDENTITY:%.*]] = "tf.Identity"([[ARG0]])
  // CHECK:      [[CLUSTER:%.*]] = "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[IDENTITY]]
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf_device.cluster"() ({
    tf_device.return %0 : tensor<i32>
  }) : () -> tensor<i32>
  // CHECK:      "tf.PartitionedCall"([[CLUSTER]])
  // CHECK-SAME: f = @callee2
  %2 = "tf.PartitionedCall"(%1) {config = "", config_proto = "", executor_type = "", f = @callee2} : (tensor<i32>) -> tensor<i32>
  func.return
}

// CHECK-LABEL: func @callee2
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func.func @callee2(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[IDENTITY:%.*]] = "tf.Identity"([[ARG0]])
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK:      return [[IDENTITY]]
  func.return %0 : tensor<i32>
}
