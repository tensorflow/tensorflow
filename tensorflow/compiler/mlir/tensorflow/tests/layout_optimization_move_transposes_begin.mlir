// RUN: tf-opt %s -tf-move-transposes=direction=begin -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @move_across_single_op
func @move_across_single_op(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%[[ARG_TRANSPOSE]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[TANH]]

  %0 = "tf.Tanh"(%arg0) : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %1 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%0, %1) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>

  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multiple_ops
func @move_across_multiple_ops(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%[[ARG_TRANSPOSE]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: %[[RELU:[0-9]*]] = "tf.Relu"(%[[TANH]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[RELU]]

  %0 = "tf.Tanh"(%arg0) : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %1 = "tf.Relu"(%0) : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>

  %2 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tf.Transpose"(%1, %2) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multi_operand_op
func @move_across_multi_operand_op(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[ARG0_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[ARG1_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg1, %[[ARG_PERM]])
  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%[[ARG0_TRANSPOSE]], %[[ARG1_TRANSPOSE]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[ADD]]

  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %1 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%0, %1) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>

  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_with_multiple_uses
func @move_with_multiple_uses(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%[[ARG_TRANSPOSE]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%[[TANH]], %[[TANH]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[ADD]]

  %0 = "tf.Tanh"(%arg0) : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %1 = "tf.AddV2"(%0, %0) : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %2 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tf.Transpose"(%1, %2) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}
