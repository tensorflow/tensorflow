// RUN: tf-opt -tf-tensor-device-copy %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @fold_identity
// CHECK-SAME: ([[arg0:%.*]]: tensor<2x2xf32>, [[arg1:%.*]]: tensor<2x2xf32>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32}} {
  func @fold_identity(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tf_executor.graph {
      // CHECK: tf.MatMul
      %outputs, %control = tf_executor.island wraps "tf.MatMul"(%arg0, %arg1) {device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      // CHECK-NOT: tf.Identity
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Identity"(%outputs) {device = ""} : (tensor<2x2xf32>) -> tensor<2x2xf32>
      tf_executor.fetch %outputs_0 : tensor<2x2xf32>
    }
    return %0 : tensor<2x2xf32>
  }
}
