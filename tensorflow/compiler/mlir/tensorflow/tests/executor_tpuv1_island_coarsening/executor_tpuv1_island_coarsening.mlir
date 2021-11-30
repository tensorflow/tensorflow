// RUN: tf-opt %s -tf-executor-tpu-v1-island-coarsening -split-input-file -verify-diagnostics | FileCheck %s

// Tests that funcs reachable from TPUPartitionedCallOps are not coarsened.
// CHECK-LABEL: func @skips_tpu_partitioned_call_reachable
func @skips_tpu_partitioned_call_reachable() {
  tf_executor.graph {
    %outputs_0, %control_1 = tf_executor.island wraps "tf.TPUOrdinalSelector"() {device = ""} : () -> tensor<?xi32>
    %control_2 = tf_executor.island wraps "tf.TPUPartitionedCall"(%outputs_0) {autotuner_thresh = 0 : i64, device = "", f = @tpu_partitioned_call_reachable} : (tensor<?xi32>) -> ()
    tf_executor.fetch
  }
  return
}

// Ensures that these islands are not coarsened (due to caller above).
// CHECK-LABEL: func @tpu_partitioned_call_reachable
func @tpu_partitioned_call_reachable() {
// CHECK-COUNT-4: island
// CHECK-NOT: island
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_1, %control_2 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %control_3 = tf_executor.island wraps "tf.OpA"() {_tpu_replicate = "cluster", f = @tpu_partitioned_call_indirectly_reachable} : () -> ()
    tf_executor.fetch
  }
  return
}

// Ensures that these islands are not coarsened (due to indirect caller above).
// CHECK-LABEL: func @tpu_partitioned_call_indirectly_reachable
func @tpu_partitioned_call_indirectly_reachable() {
// CHECK-COUNT-3: island
// CHECK-NOT: island
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %outputs_1, %control_2 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}

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

// -----

// TODO(b/188046643): Fuse op for partitioned variable within the island.

// Check that we bring in TPUPartitionedInput operand producers.
// DISABLED-CHECK-LABEL: func @fuse_in_partitioned_input_op
func @fuse_in_partitioned_input_op(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedInput"
// DISABLED-CHECK-NEXT: = "tf.AddV2"
    %outputs, %control = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01", device = "", partition_dim = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// -----

// Check that we bring in TPUPartitionedOutput users.
// DISABLED-CHECK-LABEL: func @fuse_in_partitioned_output_op
func @fuse_in_partitioned_output_op() {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.AddV2"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %outputs_3, %control_4 = tf_executor.island wraps "tf.AddV2"(%outputs_0, %outputs_0) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %partitioned_out:2, %control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%outputs_3) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tf_executor.fetch
  }
  return
}

// -----

// Check that we bring in special TPU producer ops of first island.
// CHECK-LABEL: func @fuse_in_special_tpu_operand_producer_of_first_island
func @fuse_in_special_tpu_operand_producer_of_first_island() {
  tf_executor.graph {
// CHECK: island wraps "tf.Const"
// CHECK-NEXT: island
// CHECK-NEXT: = "tf.TPUReplicatedInput"
// CHECK-NEXT: = "tf.AddV2"
    %outputs_0, %control_0 = tf_executor.island wraps "tf.Const"() {value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %replicated_out, %replicated_control = tf_executor.island wraps "tf.TPUReplicatedInput"(%outputs_0) : (tensor<4x4xf32>) -> (tensor<4x4xf32>)
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%replicated_out, %replicated_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// -----

// Check that we bring in special TPU consumer ops of first island.
// DISABLED-CHECK-LABEL: func @fuse_in_special_tpu_consumer_of_first_island
func @fuse_in_special_tpu_consumer_of_first_island() {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %partitioned_out:2, %control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%outputs_0) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tf_executor.fetch
  }
  return
}

// -----

// Check that we bring in chain of TPUReplicatedInput, TPUPartitionedInput operand producers.
// DISABLED-CHECK-LABEL: func @fuse_in_chain_special_ops_producers
func @fuse_in_chain_special_ops_producers(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedInput"
// DISABLED-CHECK-NEXT: = "tf.TPUReplicatedInput"
// DISABLED-CHECK-NEXT: = "tf.AddV2"
    %partitioned_out, %partitioned_control = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01", device = "", partition_dim = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %replicated_out, %replicated_control = tf_executor.island wraps "tf.TPUReplicatedInput"(%partitioned_out) {N = 1 : i64, T = i32, device = "", index = 0 : i64, is_mirrored_variable = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%replicated_out, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_executor.fetch
  }
  return
}

// -----

// Check that we bring in chain of TPUReplicatedOutput, TPUPartitionedOutput users.
// DISABLED-CHECK-LABEL: func @fuse_in_chain_special_ops_consumers
func @fuse_in_chain_special_ops_consumers() {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.AddV2"
// DISABLED-CHECK-NEXT: = "tf.TPUReplicatedOutput"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %add_out, %add_control = tf_executor.island wraps "tf.AddV2"(%const_out, %const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %replicated_out, %replicated_control = tf_executor.island wraps "tf.TPUReplicatedOutput"(%add_out) : (tensor<4x4xf32>) -> (tensor<4x4xf32>)
    %partitioned_out:2, %partitioned_control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%replicated_out) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tf_executor.fetch
  }
  return
}

// -----

// Check that we can bring in special TPU output ops out of order.
// DISABLED-CHECK-LABEL: func @fuse_in_special_ops_out_of_order
func @fuse_in_special_ops_out_of_order() {
  // expected-error @+1 {{or has unsupported ops}}
  tf_executor.graph {
// DISABLED-CHECK: island
// DISABLED-CHECK-NEXT: = "tf.Const"
// DISABLED-CHECK-NEXT: = "tf.SomeOp"
// DISABLED-CHECK-NEXT: = "tf.TPUReplicatedOutput"
// DISABLED-CHECK-NEXT: = "tf.TPUPartitionedOutput"
    %const_out, %const_control = tf_executor.island wraps "tf.Const"() {_tpu_replicate = "cluster", value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %some_out:2, %some_control = tf_executor.island wraps "tf.SomeOp"(%const_out) {_tpu_replicate = "cluster"} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    %partitioned_out:2, %control = tf_executor.island wraps "tf.TPUPartitionedOutput"(%some_out#1) {partition_dim = 0 : i64} : (tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    %replicated_out:2, %ireplicated_control = tf_executor.island wraps "tf.TPUReplicatedOutput"(%some_out#0) : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    tf_executor.fetch
  }
  return
}
