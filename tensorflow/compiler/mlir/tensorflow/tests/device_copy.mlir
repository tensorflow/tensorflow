// RUN: tf-opt -tf-tensor-device-copy %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @fold_identity_test
func @fold_identity_test(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tf_executor.graph {
    // CHECK: tf.MatMul
    %outputs, %control = tf_executor.island wraps "tf.MatMul"(%arg0, %arg1) {device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK-NOT: tf.Identity
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Identity"(%outputs) {device = ""} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    tf_executor.fetch %outputs_0 : tensor<2x2xf32>
  }
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @keep_identity_test
func @keep_identity_test(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tf_executor.graph {
    // CHECK: tf.MatMul
    %outputs, %control = tf_executor.island wraps "tf.MatMul"(%arg0, %arg1) {device = "/device:GPU:0", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: tf.Identity
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Identity"(%outputs) {device = "/device:CPU:0"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    tf_executor.fetch %outputs_0 : tensor<2x2xf32>
  }
  return %0 : tensor<2x2xf32>
}


// CHECK: func @while_loop_test(%[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<i32>, %arg2: tensor<*xf32>)
func @while_loop_test(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<*xf32>) {
  // CHECK-NEXT: tf.WhileRegion
  %0:2 = "tf.WhileRegion"(%arg0, %arg2) ( {
  // CHECK-NEXT: bb0(%[[ARG_3:.*]]: tensor<i32>, %[[ARG_4:.*]]: tensor<*xf32>)
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<*xf32>):
    // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.Identity"(%[[ARG_3]])
    %1 = "tf.Identity"(%arg3) : (tensor<i32>) -> tensor<i32>
    %2 = "tf.Identity"(%arg1) : (tensor<i32>) -> tensor<i32>
    // CHECK-NEXT: %[[RESULT_2:.*]] = "tf.NotEqual"(%[[RESULT_1]], %[[ARG_1]])
    %3 = "tf.NotEqual"(%1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tf.Yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<*xf32>):
    %cst = constant dense<1> : tensor<i32>
    %1 = "tf.Sub"(%arg3, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.Yield"(%1, %arg4) : (tensor<i32>, tensor<*xf32>) -> ()
  }) {is_stateless = true} : (tensor<i32>, tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>)
  return
}
