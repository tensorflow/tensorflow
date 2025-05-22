// RUN: tf-tosa-opt --tosa-fuse-bias-tf --verify-each %s | FileCheck %s


// Operations for testing --tosa-fuse-bias-tf

// ----

// CHECK-LABEL: test_conv2d_bias
// CHECK: tosa.transpose
// CHECK: tosa.conv2d
func.func @test_conv2d_bias(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<3x3x4x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32>  {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<1x4x4x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x4x4x8xf32>
  %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC"} : (tensor<1x4x4x8xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>

  func.return %1 : tensor<1x4x4x8xf32>
}
