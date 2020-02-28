// RUN: tf-opt %s -tf-layout-optimization=force-data-format=NCHW -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @transposeBiasAdd
func @transposeBiasAdd(%arg0: tensor<1x8x4x4xf32>, %arg1: tensor<8xf32>) -> tensor<1x8x4x4xf32> {

  // Convert input: NCHW -> NHWC
  %0 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x8xf32>

  // Compute in NHWC
  %2 = "tf.BiasAdd"(%1, %arg1) {data_format = "NHWC"} : (tensor<1x4x4x8xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>

  // Convert result back: NHWC -> NCHW
  %3 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %4 = "tf.Transpose"(%2, %3) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>

  // Check that BiasAdd computed in NCHW format, and all redundant transpose
  // operations removed from the function.

  // CHECK: %[[BIAS_ADD:[0-9]*]] = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW"} {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[BIAS_ADD]]

  return %4 : tensor<1x8x4x4xf32>
}