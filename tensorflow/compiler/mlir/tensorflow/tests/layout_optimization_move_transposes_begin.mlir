// RUN: tf-opt %s -tf-move-transposes=direction=begin -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @move_across_single_op
func @move_across_single_op(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
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

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
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

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
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

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
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

// CHECK-LABEL: move_transpose_handle_broadcast
func @move_transpose_handle_broadcast(%arg0:tensor<8x64xf32>, %arg1:tensor<8x64x64xf32>) -> tensor<512x64xf32> {
  %cst = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %cst_2 = "tf.Const"() {value = dense<[512, 64]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.ExpandDims"(%arg0, %cst) {device = ""} : (tensor<8x64xf32>, tensor<i32>) -> tensor<8x64x1xf32>
  %1 = "tf.AddV2"(%0, %arg1) {device = ""} : (tensor<8x64x1xf32>, tensor<8x64x64xf32>) -> tensor<8x64x64xf32>
  %2 = "tf.Transpose"(%1, %cst_1) {device = ""} : (tensor<8x64x64xf32>, tensor<3xi32>) -> tensor<64x8x64xf32>
  %3 = "tf.Reshape"(%2, %cst_2) {device = ""} : (tensor<64x8x64xf32>, tensor<2xi32>) -> tensor<512x64xf32>

  return %3 : tensor<512x64xf32>

  // CHECK: %[[CST_0:.*]] = "tf.Const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: %[[CST_1:.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[CST_2:.*]] = "tf.Const"() {value = dense<[512, 64]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[EXPAND_DIMS:.*]] = "tf.ExpandDims"(%arg0, %[[CST_1]]) {device = ""} : (tensor<8x64xf32>, tensor<i32>) -> tensor<8x64x1xf32>
  // CHECK: %[[TRANSPOSE_1:.*]] = "tf.Transpose"(%[[EXPAND_DIMS]], %[[CST_0]]) : (tensor<8x64x1xf32>, tensor<3xi32>) -> tensor<1x8x64xf32>
  // CHECK: %[[TRANSPOSE_2:.*]] = "tf.Transpose"(%arg1, %[[CST_0]]) : (tensor<8x64x64xf32>, tensor<3xi32>) -> tensor<64x8x64xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[TRANSPOSE_1]], %[[TRANSPOSE_2]]) {device = ""} : (tensor<1x8x64xf32>, tensor<64x8x64xf32>) -> tensor<64x8x64xf32>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%[[ADD]], %[[CST_2]]) {device = ""} : (tensor<64x8x64xf32>, tensor<2xi32>) -> tensor<512x64xf32>
  // CHECK: return %[[RESHAPE]] : tensor<512x64xf32>
}

// CHECK-LABEL: dont_move_transpose_different_ranks
func @dont_move_transpose_different_ranks(%arg0:tensor<1x1x2x3xf32>, %arg1:tensor<2x3xf32>) -> tensor<1x2x1x3xf32> {
  %cst = "tf.Const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<1x1x2x3xf32>, tensor<2x3xf32>) -> tensor<1x1x2x3xf32>
  %1 = "tf.Transpose"(%0, %cst) {device = ""} : (tensor<1x1x2x3xf32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>

  return %1 : tensor<1x2x1x3xf32>

  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<1x1x2x3xf32>, tensor<2x3xf32>) -> tensor<1x1x2x3xf32>
  // CHECK: %[[TRANSPOSE:.*]] = "tf.Transpose"(%[[ADD]], %[[CST]]) {device = ""} : (tensor<1x1x2x3xf32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
  // CHECK: return %[[TRANSPOSE]] : tensor<1x2x1x3xf32>
}
