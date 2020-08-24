// RUN: tf-opt %s -tf-tpu-identity-pruning | FileCheck %s --dump-input=always

// Tests Identity op in cluster is pruned away.

// CHECK-LABEL: func @testIdentity
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @testIdentity(%arg0: tensor<i32>) {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[ARG0]]
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) : () -> tensor<i32>
  return
}

// Tests IdentityN op in cluster is pruned away.

// CHECK-LABEL: func @testIdentityN
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>, [[ARG1:%.*]]: tensor<f32>)
func @testIdentityN(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  // CHECK-NOT:  "tf.IdentityN"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[ARG0]], [[ARG1]]
  %0:2 = "tf_device.cluster"() ( {
    %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    tf_device.return %1#0, %1#1 : tensor<i32>, tensor<f32>
  }) : () -> (tensor<i32>, tensor<f32>)
  return
}

// Tests transitive Identity ops reachable from the cluster are pruned away.

// CHECK-LABEL: func @testTransitiveIdentity
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @testTransitiveIdentity(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      "tf.PartitionedCall"([[ARG0]])
  // CHECK-SAME: f = @callee0
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @callee0} : (tensor<i32>) -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) : () -> tensor<i32>
  return
}

// CHECK-LABEL: func @callee0
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @callee0(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      "tf.PartitionedCall"([[ARG0]])
  // CHECK-SAME: f = @callee1
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @callee1} : (tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @callee1
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @callee1(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "tf.Identity"
  // CHECK:      return [[ARG0]]
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// Tests Identity ops not reachable from the cluster are not pruned away.

// CHECK-LABEL: func @testIdentityOutsideCluster
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @testIdentityOutsideCluster(%arg0: tensor<i32>) {
  // CHECK:      [[IDENTITY:%.*]] = "tf.Identity"([[ARG0]])
  // CHECK:      [[CLUSTER:%.*]] = "tf_device.cluster"
  // CHECK-NEXT: tf_device.return [[IDENTITY]]
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf_device.cluster"() ( {
    tf_device.return %0 : tensor<i32>
  }) : () -> tensor<i32>
  // CHECK:      "tf.PartitionedCall"([[CLUSTER]])
  // CHECK-SAME: f = @callee2
  %2 = "tf.PartitionedCall"(%1) {config = "", config_proto = "", executor_type = "", f = @callee2} : (tensor<i32>) -> tensor<i32>
  return
}

// CHECK-LABEL: func @callee2
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>)
func @callee2(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[IDENTITY:%.*]] = "tf.Identity"([[ARG0]])
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK:      return [[IDENTITY]]
  return %0 : tensor<i32>
}
