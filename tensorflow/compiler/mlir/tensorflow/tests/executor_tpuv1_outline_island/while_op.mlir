// RUN: tf-opt %s -tf-executor-tpu-v1-island-outlining | FileCheck %s --dump-input=fail

// CHECK: func @control_input
// CHECK-NOT: func @
// CHECK-LABEL: module @_tpu_v1_compat_outlined
// CHECK: @_tpu_v1_compat_outlined_func0
// CHECK: func @while_body_with_cluster_attr
// CHECK: func @while_cond_with_cluster_attr
// CHECK: func @while_body_without_cluster_attr
// CHECK: func @while_cond_without_cluster_attr
// CHECK: func @callee_func
module {
  func @control_input(%arg0: tensor<i1>) -> tensor<i32> {
    %0:4 = tf_executor.graph {
      %outputs:4, %control = tf_executor.island {
       "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
        %1 = "tf.opA"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i1>) -> tensor<i32>
        %2 = "tf.While"(%1) {body = @while_body_with_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, name = "A", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
        %3 = "tf.While"(%1) {body = @while_body_without_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, name = "C", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
        %4 = "tf.While"(%1) {body = @while_body_with_cluster_attr, cond = @while_cond_without_cluster_attr, is_stateless = false, name = "E", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
        tf_executor.yield %1, %2, %3, %4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
      }
      tf_executor.fetch %outputs#0, %outputs#1, %outputs#2, %outputs#3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>

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
  func @while_body_without_cluster_attr(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.some_op"(%arg0) : (tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
  func @while_cond_without_cluster_attr(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.PartionedCalledOp"(%arg0) { f = @callee_func} : (tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  func @callee_func(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.some_op"(%arg0) : (tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
