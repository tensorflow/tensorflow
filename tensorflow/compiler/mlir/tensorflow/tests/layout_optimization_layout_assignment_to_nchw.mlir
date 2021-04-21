// RUN: tf-opt %s -tf-layout-assignment=force-data-format=NCHW -verify-diagnostics | FileCheck %s --dump-input=always

// IMPORTANT: In the following Conv2D tests tensor shapes do not match
// convolution parameters (stride, dilations, etc...). This test only verifies
// that changing convolution data layout will update all the attributes.

// CHECK-LABEL: func @transposeConv2D
func @transposeConv2D(%input: tensor<1x32x32x3xf32>, %filter: tensor<1x1x3x8xf32>) -> tensor<1x7x7x8xf32> {

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: %[[CONV2D:[0-9]*]] = "tf.Conv2D"(%[[ARG_TRANSPOSE]], %arg1)
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: dilations = [1, 4, 2, 3]
  // CHECK-SAME: explicit_paddings = [1, 2, 7, 8, 3, 4, 5, 6]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 8, 6, 7]
  // CHECK-SAME: (tensor<1x3x32x32xf32>, tensor<1x1x3x8xf32>) -> tensor<1x8x7x7xf32>

  // CHECK: %[[RES_PERM:.*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[CONV2D]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Conv2D"(%input, %filter)
       {
         data_format = "NHWC",
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<1x32x32x3xf32>, tensor<1x1x3x8xf32>) -> tensor<1x7x7x8xf32>

  return %0 : tensor<1x7x7x8xf32>
}

// CHECK-LABEL: func @transposeConv2DWithDefaultAttr
func @transposeConv2DWithDefaultAttr(%input: tensor<1x32x32x3xf32>, %filter: tensor<1x1x3x8xf32>) -> tensor<?x?x?x?xf32>
{

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: %[[CONV2D:[0-9]*]] = "tf.Conv2D"(%[[ARG_TRANSPOSE]], %arg1)
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: dilations = [1, 4, 2, 3]
  // CHECK-SAME: explicit_paddings = [1, 2, 7, 8, 3, 4, 5, 6]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 8, 6, 7]
  // CHECK-SAME: (tensor<1x3x32x32xf32>, tensor<1x1x3x8xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: %[[RES_PERM:.*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[CONV2D]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  // (1) data_format attribute has default value NHWC
  // (2) result shape is unknown (check that optimizer does not fail)
  %0 = "tf.Conv2D"(%input, %filter)
       {
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<1x32x32x3xf32>, tensor<1x1x3x8xf32>) -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @transposeConv2DBackpropFilter
func @transposeConv2DBackpropFilter(
  %input: tensor<1x32x32x3xf32>,
  %filter_sizes: tensor<4xi32>,
  %out_backprop: tensor<1x32x32x8xf32>
) -> tensor<1x1x3x8xf32> {

  // CHECK: %[[FILTER_PERM:[0-9]*]] = "tf.DataFormatVecPermute"
  // CHECK-SAME: dst_format = "NCHW"
  // CHECK-SAME: src_format = "NHWC"

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[IN_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[OUT_BP_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg2, %[[ARG_PERM]])

  // CHECK: %[[CONV2D_BACKPROP:[0-9]*]] = "tf.Conv2DBackpropFilter"
  // CHECK-SAME: (%[[IN_TRANSPOSE]], %[[FILTER_PERM]], %[[OUT_BP_TRANSPOSE]])
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: dilations = [1, 4, 2, 3]
  // CHECK-SAME: explicit_paddings = [1, 2, 7, 8, 3, 4, 5, 6]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 8, 6, 7]
  // CHECK-SAME: (tensor<1x3x32x32xf32>, tensor<4xi32>, tensor<1x8x32x32xf32>)
  // CHECK-SAME: -> tensor<1x1x3x8xf32>

  // CHECK: return %[[CONV2D_BACKPROP]]

  %0 = "tf.Conv2DBackpropFilter"(%input, %filter_sizes, %out_backprop)
       {
         data_format = "NHWC",
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<1x32x32x3xf32>, tensor<4xi32>, tensor<1x32x32x8xf32>)
         -> tensor<1x1x3x8xf32>

  return %0 : tensor<1x1x3x8xf32>
}

// CHECK-LABEL: func @transposeConv2DBackpropInput
func @transposeConv2DBackpropInput(
  %input_sizes: tensor<4xi32>,
  %filter: tensor<1x1x3x8xf32>,
  %out_backprop: tensor<1x32x32x8xf32>
) -> tensor<1x32x32x3xf32> {

  // CHECK: %[[INPUT_PERM:[0-9]*]] = "tf.DataFormatVecPermute"
  // CHECK-SAME: dst_format = "NCHW"
  // CHECK-SAME: src_format = "NHWC"

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[OUT_BP_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg2, %[[ARG_PERM]])

  // CHECK: %[[CONV2D_BACKPROP:[0-9]*]] = "tf.Conv2DBackpropInput"
  // CHECK-SAME: (%[[INPUT_PERM]], %arg1, %[[OUT_BP_TRANSPOSE]])
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: dilations = [1, 4, 2, 3]
  // CHECK-SAME: explicit_paddings = [1, 2, 7, 8, 3, 4, 5, 6]
  // CHECK-SAME: padding = "EXPLICIT"
  // CHECK-SAME: strides = [5, 8, 6, 7]
  // CHECK-SAME: (tensor<4xi32>, tensor<1x1x3x8xf32>, tensor<1x8x32x32xf32>)
  // CHECK-SAME: -> tensor<1x3x32x32xf32>

  // CHECK: %[[RES_PERM:.*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[CONV2D_BACKPROP]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Conv2DBackpropInput"(%input_sizes, %filter, %out_backprop)
       {
         data_format = "NHWC",
         dilations = [1, 2, 3, 4],
         explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8],
         padding = "EXPLICIT",
         strides = [5, 6, 7, 8]
       } : (tensor<4xi32>, tensor<1x1x3x8xf32>, tensor<1x32x32x8xf32>)
         -> tensor<1x32x32x3xf32>

  return %0 : tensor<1x32x32x3xf32>
}

// CHECK-LABEL: func @transposeFusedBatchNormV3
func @transposeFusedBatchNormV3(
  %arg0: tensor<1x28x28x64xf32>,
  %arg1: tensor<64xf32>
) -> tensor<1x28x28x64xf32> {

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}
  // CHECK: %[[ARG_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])

  // CHECK: "tf.FusedBatchNormV3"
  // CHECK-SAME: (%[[ARG_TRANSPOSE]], %arg1, %arg1, %arg1, %arg1)
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: (tensor<1x64x28x28xf32>, tensor<64xf32>,
  // CHECK-SAME: -> (tensor<1x64x28x28xf32>, tensor<64xf32>,

  // CHECK: %[[RES_PERM:.*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%y, %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %y, %batch_mean, %batch_var, %reserve_1, %reserve_2, %reserve_3
    = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg1, %arg1, %arg1)
       {
         data_format = "NHWC",
         epsilon = 1.001 : f32,
         exponential_avg_factor = 1.0 : f32,
         is_training = true
       }
        : (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64xf32>)
       -> (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)

  return %y : tensor<1x28x28x64xf32>
}

// CHECK-LABEL: func @transposeFusedBatchNormGradV3
func @transposeFusedBatchNormGradV3(
  %arg0: tensor<1x28x28x64xf32>,
  %arg1: tensor<1x28x28x64xf32>,
  %arg2: tensor<64xf32>
) -> tensor<1x28x28x64xf32> {

  // CHECK: %[[ARG_PERM:.*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 3, 1, 2]> : tensor<4xi64>}

  // CHECK: %[[ARG0_TPOSE:[0-9]*]] = "tf.Transpose"(%arg0, %[[ARG_PERM]])
  // CHECK: %[[ARG1_TPOSE:[0-9]*]] = "tf.Transpose"(%arg1, %[[ARG_PERM]])

  // CHECK: "tf.FusedBatchNormGradV3"
  // CHECK-SAME: (%[[ARG0_TPOSE]], %[[ARG1_TPOSE]], %arg2, %arg2, %arg2, %arg2)
  // CHECK-SAME: data_format = "NCHW"
  // CHECK-SAME: (tensor<1x64x28x28xf32>, tensor<1x64x28x28xf32>,
  // CHECK-SAME: -> (tensor<1x64x28x28xf32>,

  // CHECK: %[[RES_PERM:.*]] = "tf.Const"()
  // CHECK-SAME: {value = dense<[0, 2, 3, 1]> : tensor<4xi64>}

  // CHECK: %[[RES_TPOSE:[0-9]*]] = "tf.Transpose"
  // CHECK-SAME: (%x_backprop, %[[RES_PERM]])
  // CHECK: return %[[RES_TPOSE]]

  %x_backprop, %scale_backprop, %offset_backprop, %reserve_1, %reserve_2
    = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg2, %arg2, %arg2)
       {
         data_format = "NHWC",
         epsilon = 1.001 : f32,
         exponential_avg_factor = 1.0 : f32,
         is_training = true
       }
        : (tensor<1x28x28x64xf32>, tensor<1x28x28x64xf32>,
           tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
       -> (tensor<1x28x28x64xf32>,
           tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)

  return %x_backprop : tensor<1x28x28x64xf32>
}
