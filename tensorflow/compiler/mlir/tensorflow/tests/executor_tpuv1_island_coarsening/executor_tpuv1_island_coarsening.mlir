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
