// RUN: tf-opt -split-input-file -verify-diagnostics -tfl-get-arithmetic-count %s | FileCheck %s

func @testConv2D(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: _arithmetic_count = 230686720 : i64
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  return %0 : tensor<256x32x32x16xf32>
}

func @testConv2DDynamicShape(tensor<?x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32> {
^bb0(%arg0: tensor<?x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: _arithmetic_count = -1 : i64
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<?x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32>
  return %0 : tensor<?x32x32x16xf32>
}
