// RUN: tf-opt %s -tf-layout-assignment=force-data-format=NCHW -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @transposeBiasAdd
func @transposeBiasAdd(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<1x4x4x8xf32> {

  // Check that BiasAdd was converted to forced data format, and layout
  // dependent arguments and results passed through transpose nodes.

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[BIAS_ADD:[0-9]*]] = "tf.BiasAdd"(%[[ARG_TRANSPOSE]], %arg1) {data_format = "NCHW"} {{.*}} tensor<1x8x4x4xf32>
  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[BIAS_ADD]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<1x4x4x8xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>

  return %0 : tensor<1x4x4x8xf32>
}

// CHECK-LABEL: func @transposeBiasAddWithDefaultAttr
func @transposeBiasAddWithDefaultAttr(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<1x4x4x8xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[BIAS_ADD:[0-9]*]] = "tf.BiasAdd"(%[[ARG_TRANSPOSE]], %arg1) {data_format = "NCHW"} {{.*}} tensor<1x8x4x4xf32>
  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[BIAS_ADD]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) : (tensor<1x4x4x8xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>

  return %0 : tensor<1x4x4x8xf32>
}

// CHECK-LABEL: func @transposeBiasWithUnknownShape
func @transposeBiasWithUnknownShape(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<*xf32> {

  // CHECK: %[[BIAS_ADD:[0-9]*]] = "tf.BiasAdd"(%[[ARG_TRANSPOSE]], %arg1) {data_format = "NCHW"} {{.*}} tensor<*xf32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) : (tensor<1x4x4x8xf32>, tensor<8xf32>) -> tensor<*xf32>

  return %0 : tensor<*xf32>
}