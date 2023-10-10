// RUN: tf-opt -tfl-prepare-tf=allow-bf16-and-f16-type-legalization=true %s | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: conv_2d_bf16
func.func @conv_2d_bf16(%arg0 : tensor<256x32x32x3xbf16>, %arg1 : tensor<3x3x3x16xbf16>) -> tensor<256x8x7x16xbf16> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xbf16>, tensor<3x3x3x16xbf16>) -> tensor<256x8x7x16xbf16>
  func.return %0 : tensor<256x8x7x16xbf16>
  // CHECK: "tfl.conv_2d"
}

// CHECK-LABEL: fused_batch_norm_v3_bf16
func.func @fused_batch_norm_v3_bf16(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x8x8x8xbf16> {
  %0, %1, %2 ,%3, %4, %5 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_BFLOAT16", U = "tfdtype$DT_BFLOAT16", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0 : tensor<8x8x8x8xbf16>
  // CHECK: "tf.FusedBatchNormV3"
}

// CHECK-LABEL: depthwise_conv_2d_bf16
func.func @depthwise_conv_2d_bf16(%arg0 : tensor<256x32x32x3xbf16>, %arg1 : tensor<3x3x3x4xf32>, %arg2 : tensor<256x3x32x32xf32>) -> tensor<256x30x30x12xbf16> {
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xbf16>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xbf16>
  func.return %0 : tensor<256x30x30x12xbf16>
  // CHECK: "tfl.depthwise_conv_2d"
}

// CHECK-LABEL: conv_2d_f16
func.func @conv_2d_f16(%arg0 : tensor<256x32x32x3xf16>, %arg1 : tensor<3x3x3x16xf16>) -> tensor<256x8x7x16xf16> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf16>, tensor<3x3x3x16xf16>) -> tensor<256x8x7x16xf16>
  func.return %0 : tensor<256x8x7x16xf16>
  // CHECK: "tfl.conv_2d"
}

}
