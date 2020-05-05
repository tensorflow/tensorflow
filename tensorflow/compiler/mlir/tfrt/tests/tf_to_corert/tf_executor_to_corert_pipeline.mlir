// RUN: tf-opt -tf-executor-to-corert-pipeline %s | FileCheck %s

// CHECK-LABEL: func @basic
// CHECK-SAME: ([[arg0:%.*]]: !corert.tensorhandle, [[arg1:%.*]]: !corert.tensorhandle)
// CHECK-NEXT: [[cpu:%.*]] = corert.get_device "cpu"
// CHECK-NEXT: [[res:%.*]] = corert.executeop([[cpu]]) "tf.MatMul"([[arg0]], [[arg1]])
// CHECK-NEXT: hex.return [[res]] : !corert.tensorhandle
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 293 : i32}} {
  func @basic(%arg0: tensor<3x1xf32>,
              %arg1: tensor<!tf.resource<tensor<1x3xf32>>>
  ) -> tensor<3x3xf32> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<0.899999976> : tensor<f32>} : () -> tensor<f32>
      %outputs_0, %control_0 = tf_executor.island {
        %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf.resource<tensor<1x3xf32>>>) -> tensor<*x!tf.resource>
        %2 = "tf.ReadVariableOp"(%1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "", dtype = f32} : (tensor<*x!tf.resource>) -> tensor<1x3xf32>
        %3 = "tf.MatMul"(%arg0, %2) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
        tf_executor.yield %3 : tensor<3x3xf32>
      }
      tf_executor.fetch %outputs_0, %control_0 : tensor<3x3xf32>, !tf_executor.control
    }
    return %0 : tensor<3x3xf32>
  }
}
