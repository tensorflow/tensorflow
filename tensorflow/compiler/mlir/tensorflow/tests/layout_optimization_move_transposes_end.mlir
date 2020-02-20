// RUN: tf-opt %s -tf-move-transposes=direction=end -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @move_across_single_op
func @move_across_single_op(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%arg0) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[TANH]], %[[RES_PERM]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Tanh"(%1) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multiple_ops
func @move_across_multiple_ops(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[TANH0:[0-9]*]] = "tf.Tanh"(%arg0) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[TANH1:[0-9]*]] = "tf.Tanh"(%[[TANH0]]) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[TANH1]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Tanh"(%1) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Tanh"(%2) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multi_operand_op
func @move_across_multi_operand_op(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%arg0, %arg1) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[ADD]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Transpose"(%arg1, %0) : (tensor<1x4x4x8xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
  %3 = "tf.AddV2"(%1, %2) : (tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}
