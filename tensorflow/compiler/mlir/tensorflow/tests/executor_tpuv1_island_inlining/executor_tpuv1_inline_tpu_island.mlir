// RUN: tf-opt %s -tf-executor-tpu-v1-island-inlining | FileCheck %s

// Check that the nested module is inlined and erased.

module {
// CHECK-LABEL: func @func0
  func @func0(%arg0: tensor<i1>) -> tensor<f32> {
    %0 = tf_executor.graph {
// CHECK-NOT: PartitionedCall
// CHECK: "tf.opA"
      %outputs, %control = tf_executor.island wraps "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0} : (tensor<i1>) -> tensor<i1>
      %outputs_0, %control_1 = tf_executor.island(%control) {
        %1 = "tf.opB"() : () -> tensor<f32>
        tf_executor.yield %1 : tensor<f32>
      }
      tf_executor.fetch %outputs_0 : tensor<f32>
    }
    return %0 : tensor<f32>
  }
// CHECK-LABEL: func @func2
  func @func2(%arg0: tensor<i1>) -> tensor<i1> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        %1 = "tf.opB"() : () -> tensor<f32>
        tf_executor.yield %1 : tensor<f32>
      }
// CHECK-NOT: PartitionedCall
// CHECK: "tf.opA"
// CHECK: "tf.opA"
// CHECK: "tf.SomeOp"
      %outputs_0:2, %control_1 = tf_executor.island wraps "tf.PartitionedCall"(%arg0, %outputs) {config = "", config_proto = "", executor_type = "", f = @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func1} : (tensor<i1>, tensor<f32>) -> (tensor<i1>, tensor<i32>)
      tf_executor.fetch %outputs_0#0 : tensor<i1>
    }
    return %0 : tensor<i1>
  }
// CHECK-NOT: _tpu_v1_compat_outlined
  module @_tpu_v1_compat_outlined {
    func nested @_tpu_v1_compat_outlined_func0(%arg0: tensor<i1>) -> tensor<i1> {
      %0 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      return %0 : tensor<i1>
    }
    func nested @_tpu_v1_compat_outlined_func1(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<i1>, tensor<i32>) {
      %0 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %1 = "tf.opA"(%0) : (tensor<i1>) -> tensor<i1>
      %2 = "tf.SomeOp"(%arg0, %arg1) : (tensor<i1>, tensor<f32>) -> tensor<i32>
      return %1, %2 : tensor<i1>, tensor<i32>
    }
  }
}
