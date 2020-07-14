// RUN: tf-opt %s -tf-executor-tpu-v1-island-inlining | FileCheck %s

// CHECK-NOT: tf.PartitionedCall
// CHECK-NOT: module @_tpu_v1_compat_outlined

module {
  func @control_input(%arg0: tensor<i1>) -> tensor<i32> {
    %0:4 = tf_executor.graph {
      %outputs:4, %control = tf_executor.island wraps "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0} : (tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)
      tf_executor.fetch %outputs#0, %outputs#1, %outputs#2, %outputs#3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
    }
    return %0#0 : tensor<i32>
  }
  module @_tpu_v1_compat_outlined {
    func @_tpu_v1_compat_outlined_func0(%arg0: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
      "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "device", num_replicas = 1 : i64, topology = "topology"} : () -> ()
      %0 = "tf.opA"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i1>) -> tensor<i32>
      %1 = "tf.While"(%0) {body = @while_body_with_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, name = "A", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
      %2 = "tf.While"(%0) {body = @while_body_without_cluster_attr, cond = @while_cond_with_cluster_attr, is_stateless = false, name = "C", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
      %3 = "tf.While"(%0) {body = @while_body_with_cluster_attr, cond = @while_cond_without_cluster_attr, is_stateless = false, name = "E", parallel_iterations = 10 : i64} : (tensor<i32>) -> tensor<i32>
      return %0, %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
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
      %0 = "tf.PartionedCalledOp"(%arg0) {f = @callee_func} : (tensor<i32>) -> tensor<i1>
      return %0 : tensor<i1>
    }
    func @callee_func(%arg0: tensor<i32>) -> tensor<i1> {
      %0 = "tf.some_op"(%arg0) : (tensor<i32>) -> tensor<i1>
      return %0 : tensor<i1>
    }
  }
}
