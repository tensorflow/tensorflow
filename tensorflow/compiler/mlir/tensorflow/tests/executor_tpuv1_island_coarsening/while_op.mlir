// RUN: tf-opt %s -tf-executor-tpu-v1-island-coarsening | FileCheck %s --dump-input=fail


// Test that islands with a function call are merged if the call is to a function
// that contains ops with the same attribute.
// CHECK-LABEL: func @control_input
func @control_input(%arg0 : tensor<i1>) -> tensor<i32> {
  %0:6 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.opA"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i1>) -> tensor<i32>
    %2:2 = tf_executor.island wraps "tf.While"(%1#0) {name = "A", body = @while_body_with_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
    %3:2 = tf_executor.island wraps "tf.While"(%1#0) {name = "B", body = @while_body_with_wrong_cluster_attr, cond = @while_cond_with_wrong_cluster_attr, is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
    %4:2 = tf_executor.island wraps "tf.While"(%1#0) {name = "C", body = @while_body_without_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
    %6:2 = tf_executor.island wraps "tf.While"(%1#0) {name = "D", body = @while_body_without_cluster_attr, cond = @while_cond_without_cluster_attr, is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
    %5:2 = tf_executor.island wraps "tf.While"(%1#0) {name = "E", body = @while_body_with_cluster_attr, cond = @while_cond_without_cluster_attr, is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>

// CHECK: "tf.opA"
// CHECK-NOT: island
// CHECK: name = "A"
// CHECK-NOT: island
// CHECK: name = "C"
// CHECK-NOT: island
// CHECK: name = "E"
// CHECK: island {{.*}}name = "B"
// CHECK: island {{.*}}name = "D"

    tf_executor.fetch %1#0, %2#0, %3#0, %4#0, %5#0, %6#0 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }
  return %0#0 : tensor<i32>
}

func @while_body_with_cluster_attr(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.some_op"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}
func @while_cond_with_cluster_attr(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.some_op"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func @while_body_with_wrong_cluster_attr(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.some_op"(%arg0) {_tpu_replicate = "wrong_cluster"} : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}
func @while_cond_with_wrong_cluster_attr(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.some_op"(%arg0) {_tpu_replicate = "wrong_cluster"} : (tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func @while_body_without_cluster_attr(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.some_op"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}
func @while_cond_without_cluster_attr(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.some_op"(%arg0) : (tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

