// RUN: tf-opt %s -tf-executor-tpu-v1-island-coarsening | FileCheck %s


// Test that islands without the attribute are not merged.
// CHECK-LABEL: func @control_input
func @control_input(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }

// CHECK: "tf.opA"
// CHECK: island
// CHECK: "tf.opB"

    tf_executor.fetch %2#0 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// Check that we fuse entirely when the attribute matches.
// CHECK-LABEL: func @all_fused
func @all_fused(%arg0: tensor<*xf32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
    %outputs, %control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}


// Check that we don't fuse an op that does not have the attribute.
// CHECK-LABEL: func @split_ops
func @split_ops(%arg0: tensor<*xf32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK-SAME: _tpu_replicate
// CHECK: island wraps "tf.AddV2"
    %outputs, %control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %outputs_5, %control_6 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}


// Check that we correctly merge operations from two clusters in their
// respective clusters.
// CHECK-LABEL: func @two_clusters_mixed
func @two_clusters_mixed(%arg0: tensor<*xf32>) {
  tf_executor.graph {
// CHECK: %[[ISLAND1:.*]], {{.*}} = tf_executor.island
// CHECK-NEXT: = "tf.Const"{{.*}}"cluster1"
// CHECK-NEXT: = "tf.Const"{{.*}}"cluster1"
// CHECK-NEXT: = "tf.AddV2"{{.*}}"cluster1"
// CHECK-NEXT: = "tf.AddV2"{{.*}}"cluster1"
// CHECK: %[[ISLAND2:.*]], {{.*}} = tf_executor.island
// CHECK-NEXT: = "tf.Const"{{.*}}"cluster2"
// CHECK-NEXT: = "tf.Const"{{.*}}"cluster2"
// CHECK-NEXT: = "tf.AddV2"{{.*}}"cluster2"
// CHECK-NEXT: = "tf.AddV2"{{.*}}"cluster2"
// CHECK: island wraps "tf.AddV2"(%[[ISLAND1]], %[[ISLAND2]])
    %outputs, %control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster1", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster2", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_2, %control_2 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster1", value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %outputs_3, %control_3 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster2", value = dense<4> : tensor<i32>} : () -> tensor<i32>
    %outputs_4, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %outputs_5, %control_5 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_2) {_tpu_replicate = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %outputs_6, %control_6 = tf_executor.island wraps "tf.AddV2"(%outputs_0, %outputs_3) {_tpu_replicate = "cluster2"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %outputs_7, %control_7 = tf_executor.island wraps "tf.AddV2"(%outputs_5, %outputs) {_tpu_replicate = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %outputs_8, %control_8 = tf_executor.island wraps "tf.AddV2"(%outputs_0, %outputs_3) {_tpu_replicate = "cluster2"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}


// Check that we bring in TPUReplicatedInputOp operand producers.
// CHECK-LABEL: func @fuse_in_replicated_input_op
func @fuse_in_replicated_input_op(%arg0: tensor<i32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.TPUReplicatedInput"
// CHECK-NEXT: = "tf.AddV2"
    %outputs, %control = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0) {N = 1 : i64, T = i32, device = "", index = 0 : i64, is_mirrored_variable = false} : (tensor<i32>) -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}


// Check that we bring in TPUReplicatedOutputOp users.
// CHECK-LABEL: func @fuse_in_replicated_output_op
func @fuse_in_replicated_output_op() {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK-NEXT: = "tf.TPUReplicatedOutput"
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs_0, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %replicated_out, %control = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_3) : (tensor<i32>) -> (tensor<i32>)
    tf_executor.fetch
  }
  return
}

// Check that we bring in TPUPartitionedInput operand producers.
// CHECK-LABEL: func @fuse_in_partitioned_input_op
func @fuse_in_partitioned_input_op(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.TPUPartitionedInput"
// CHECK-NEXT: = "tf.AddV2"
    %outputs, %control = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01", device = "", partition_dim = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// Check that we bring in TPUPartitionedOutput users.
// CHECK-LABEL: func @fuse_in_partitioned_output_op
func @fuse_in_partitioned_output_op() {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs_0, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %partitioned_out:2, %control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%outputs_3) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tf_executor.fetch
  }
  return
}

// Check that we bring in chain of TPUReplicatedInput, Identity and TPUPartitionedInput operand producers.
// CHECK-LABEL: func @fuse_in_chain_special_ops_producers
func @fuse_in_chain_special_ops_producers(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.TPUPartitionedInput"
// CHECK-NEXT: = "tf.Identity"
// CHECK-NEXT: = "tf.TPUReplicatedInput"
// CHECK-NEXT: = "tf.AddV2"
    %partitioned_out, %partitioned_control = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01", device = "", partition_dim = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %identity_out, %identity_control = tf_executor.island wraps "tf.Identity"(%partitioned_out) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %replicated_out, %replicated_control = tf_executor.island wraps "tf.TPUReplicatedInput"(%identity_out) {N = 1 : i64, T = i32, device = "", index = 0 : i64, is_mirrored_variable = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%replicated_out, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// Check that we bring in chain of TPUReplicatedOutput, Identity and TPUPartitionedOutput users.
// CHECK-LABEL: func @fuse_in_chain_special_ops_consumers
func @fuse_in_chain_special_ops_consumers() {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK-NEXT: = "tf.TPUReplicatedOutput"
// CHECK-NEXT: = "tf.Identity"
// CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%const_out, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %replicated_out, %replicated_control = tf_executor.island wraps "tf.TPUReplicatedOutput"(%add_out) : (tensor<4x4xf32>) -> (tensor<4x4xf32>)
    %identity_out, %identity_control = tf_executor.island wraps "tf.Identity"(%replicated_out) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %partitioned_out:2, %partitioned_control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%identity_out) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tf_executor.fetch
  }
  return
}

// Check that we can bring in special TPU output ops out of order.
// CHECK-LABEL: func @fuse_in_special_ops_out_of_order
func @fuse_in_special_ops_out_of_order() {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.SomeOp"
// CHECK-NEXT: = "tf.TPUReplicatedOutput"
// CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %some_out:2, %some_control = tf_executor.island wraps "tf.SomeOp"(%const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    %partitioned_out:2, %control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%some_out#1) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    %replicated_out:2, %ireplicated_control = tf_executor.island wraps "tf.TPUReplicatedOutput"(%some_out#0) : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    tf_executor.fetch
  }
  return
}

// Check that we do not fuse identity producers with use outside cluster
// CHECK-LABEL: func @do_not_fuse_identity_with_outside_use
func @do_not_fuse_identity_with_outside_use(%arg0: tensor<4x4xf32>) {
  tf_executor.graph {
// CHECK: island wraps "tf.Identity"
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
    %identity_out, %identity_control = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%identity_out, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %mul_out, %mul_control = tf_executor.island wraps "tf.Mul"(%identity_out, %identity_out) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// Check that we do not fuse IdentityN consumers with operands outside cluster
// CHECK-LABEL: func @do_not_fuse_identityN_with_outside_operand
func @do_not_fuse_identityN_with_outside_operand(%arg0: tensor<4x4xf32>) {
  tf_executor.graph {
// CHECK: island wraps "tf.Mul"
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK: island wraps "tf.IdentityN"
    %mul_out, %mul_control = tf_executor.island wraps "tf.Mul"(%arg0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%arg0, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %identityN_out:2, %identity_control = tf_executor.island wraps "tf.IdentityN"(%add_out, %mul_out) : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    tf_executor.fetch
  }
  return
}

// Check that we do not fuse identity with different cluster attribute
// CHECK-LABEL: func @do_not_fuse_identity_with_different_cluster
func @do_not_fuse_identity_with_different_cluster(%arg0: tensor<4x4xf32>) {
  tf_executor.graph {
// CHECK: island
// CHECK-NEXT: = "tf.Const"
// CHECK-NEXT: = "tf.AddV2"
// CHECK: island wraps "tf.Identity"
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster_1", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%arg0, %const_out) {_tpu_replicate = "cluster_1"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %identity_out, %identity_control = tf_executor.island wraps "tf.Identity"(%add_out) {_tpu_replicate = "cluster_0"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}
