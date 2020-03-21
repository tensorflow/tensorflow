// RUN: tf-opt %s -tf-layout-assignment=force-data-format=NHWC -verify-diagnostics | FileCheck %s --dump-input=always

// IMPORTANT: Tensor shapes do not match convolution parameters (stride,
// dilations, etc...). This test only verifies that changing convolution data
// layout will update all the attributes.

// CHECK-LABEL: func @transposeConv2D
func @transposeConv2D(%input: tensor<1x3x32x32xf32>, %filter: tensor<1x1x3x8xf32>) -> tensor<1x8x32x32xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: %[[CONV2D:[0-9]*]] = "tf.Conv2D"(%[[ARG_TRANSPOSE]], %arg1)
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: dilations = [1, 3, 4, 2]
  // CHECK-SAME: explicit_paddings = [1, 2, 5, 6, 7, 8, 3, 4]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 7, 8, 6]
  // CHECK-SAME: (tensor<1x32x32x3xf32>, tensor<1x1x3x8xf32>) -> tensor<1x32x32x8xf32>

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[CONV2D]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Conv2D"(%input, %filter)
       {
         data_format = "NCHW",
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<1x3x32x32xf32>, tensor<1x1x3x8xf32>) -> tensor<1x8x32x32xf32>

  return %0 : tensor<1x8x32x32xf32>
}
