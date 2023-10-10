// RUN: tf-opt %s -tf-layout-optimization=force-data-format=NCHW -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @transposeConv2D
func.func @transposeConv2D(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<1x1x3x8xf32>) -> tensor<1x8x32x32xf32> {

  // Convert input: NCHW -> NHWC
  %0 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x3x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x3xf32>

  // Compute in NHWC
  %2 = "tf.Conv2D"(%1, %arg1)
    {
      data_format = "NHWC",
      padding = "SAME",
      strides = [1, 1, 1, 1],
      dilations = [1, 1, 1, 1]
    } : (tensor<1x32x32x3xf32>, tensor<1x1x3x8xf32>) -> tensor<1x32x32x8xf32>

  // Convert result back: NHWC -> NCHW
  %3 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %4 = "tf.Transpose"(%2, %3) : (tensor<1x32x32x8xf32>, tensor<4xi32>) -> tensor<1x8x32x32xf32>

  // Check that Conv2D computed in NCHW format, and all redundant transpose
  // operations removed from the function.

  // CHECK: %[[CONV:[0-9]*]] = "tf.Conv2D"(%arg0, %arg1)
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: -> tensor<1x8x32x32xf32>

  // CHECK: return %[[CONV]]

  func.return %4 : tensor<1x8x32x32xf32>
}