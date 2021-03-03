// RUN: tf-opt %s -tf-layout-assignment=force-data-format=NHWC -verify-diagnostics | FileCheck %s --dump-input=always

// IMPORTANT: Tensor shapes do not match convolution parameters (stride,
// dilations, etc...). This test only verifies that changing convolution data
// layout will update all the attributes.

// CHECK-LABEL: func @transposeConv2D
func @transposeConv2D(%input: tensor<1x3x32x32xf32>, %filter: tensor<1x1x3x8xf32>) -> tensor<1x8x7x6xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: %[[CONV2D:[0-9]*]] = "tf.Conv2D"(%[[ARG_TRANSPOSE]], %arg1)
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: dilations = [1, 3, 4, 2]
  // CHECK-SAME: explicit_paddings = [1, 2, 5, 6, 7, 8, 3, 4]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 7, 8, 6]
  // CHECK-SAME: (tensor<1x32x32x3xf32>, tensor<1x1x3x8xf32>) -> tensor<1x7x6x8xf32>

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[CONV2D]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Conv2D"(%input, %filter)
       {
         data_format = "NCHW",
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<1x3x32x32xf32>, tensor<1x1x3x8xf32>) -> tensor<1x8x7x6xf32>

  return %0 : tensor<1x8x7x6xf32>
}

// CHECK-LABEL: func @transposeFusedBatchNormV3
func @transposeFusedBatchNormV3(
  %arg0: tensor<1x64x28x28xf32>,
  %arg1: tensor<64xf32>
) -> tensor<1x64x28x28xf32> {

  // CHECK: %[[ARG_PERM:[0-9]*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: "tf.FusedBatchNormV3"
  // CHECK-SAME: (%[[ARG_TRANSPOSE]], %arg1, %arg1, %arg1, %arg1)
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: (tensor<1x28x28x64xf32>, tensor<64xf32>,
  // CHECK-SAME: -> (tensor<1x28x28x64xf32>, tensor<64xf32>,

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%y, %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %y, %batch_mean, %batch_var, %reserve_1, %reserve_2, %reserve_3
    = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg1, %arg1, %arg1)
       {
         data_format = "NCHW",
         epsilon = 1.001 : f32,
         exponential_avg_factor = 1.0 : f32,
         is_training = true
       }
        : (tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64xf32>)
       -> (tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)

  return %y : tensor<1x64x28x28xf32>
}

// CHECK-LABEL: bias_add_nchw
func @bias_add_nchw(%arg0: tensor<1x256x150x150xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x150x150xf32> {
  // CHECK: (%[[ARG0:.*]]: tensor<1x256x150x150xf32>, %[[ARG1:.*]]: tensor<256xf32>)
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[R0:.*]] = "tf.Transpose"(%[[ARG0]], %[[CST]])
  // CHECK: %[[R1:.*]] = "tf.BiasAdd"(%[[R0]], %[[ARG1]]) {data_format = "NHWC", device = ""}
  // CHECK: %[[CST_0:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: "tf.Transpose"(%[[R1]], %[[CST_0]])
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW", device = ""} : (tensor<1x256x150x150xf32>, tensor<256xf32>) -> tensor<1x256x150x150xf32>
  return %0 : tensor<1x256x150x150xf32>
}

// CHECK-LABEL: maxpool_nchw
func @maxpool_nchw(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32> {
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[R0:.*]] = "tf.Transpose"(%arg0, %[[CST]])
  // CHECK: %[[R1:.*]] = "tf.MaxPool"(%[[R0]]) {data_format = "NHWC", explicit_paddings = [], ksize = [1, 3, 3, 1], padding = "SAME", strides = [1, 2, 2, 1]}
  // CHECK: %[[CST_0:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: "tf.Transpose"(%[[R1]], %[[CST_0]])
  %0 = "tf.MaxPool"(%arg0)
       {
         data_format = "NCHW",
         ksize = [1, 1, 3, 3],
         padding = "SAME",
         strides = [1, 1, 2, 2]
       } : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
  return %0 : tensor<1x64x56x56xf32>
}
