// RUN: tf-tosa-opt --split-input-file --tosa-legalize-tfl --verify-diagnostics %s | FileCheck %s

// -----
// CHECK-LABEL: max_pool_same_padding_dyn
// CHECK: tfl.max_pool_2d
func.func @max_pool_same_padding_dyn(%arg0: tensor<?x1x?x23xf32>) -> (tensor<?x1x?x23xf32>) {
%0 = "tfl.max_pool_2d"(%arg0) <{filter_height = 1 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 4 : i32}> : (tensor<?x1x?x23xf32>) -> tensor<?x1x?x23xf32>
return %0 : tensor<?x1x?x23xf32>
}

// -----
// CHECK-LABEL: avg_pool_same_padding_dyn
// CHECK: tfl.average_pool_2d
func.func @avg_pool_same_padding_dyn(%arg0: tensor<?x1x?x23xf32>) -> (tensor<?x1x?x23xf32>) {
%0 = "tfl.average_pool_2d"(%arg0) <{filter_height = 1 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 4 : i32}> : (tensor<?x1x?x23xf32>) -> tensor<?x1x?x23xf32>
return %0 : tensor<?x1x?x23xf32>
}

// -----
// CHECK-LABEL: conv2d_same_padding_dyn
// CHECK: tfl.conv_2d
func.func @conv2d_same_padding_dyn(%input: tensor<1x?x32x8xf32>, %filter: tensor<5x3x3x8xf32>, %bias: tensor<5xf32>) -> tensor<1x32x32x5xf32> {
  %0 = "tfl.conv_2d"(%input, %filter, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 1 : i32} : (tensor<1x?x32x8xf32>, tensor<5x3x3x8xf32>, tensor<5xf32>) -> tensor<1x32x32x5xf32>
  return %0 : tensor<1x32x32x5xf32>
}

// -----
// CHECK-LABEL: conv3d_same_padding_dyn
// CHECK: tfl.conv_3d
func.func @conv3d_same_padding_dyn(%arg0: tensor<2x2x7x?x2xf32>, %arg1: tensor<2x3x3x2x4xf32>) -> tensor<2x2x7x7x4xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 4 : i32} : (tensor<2x2x7x?x2xf32>, tensor<2x3x3x2x4xf32>, none) -> tensor<2x2x7x7x4xf32>
  func.return %0 : tensor<2x2x7x7x4xf32>
}

// -----
// CHECK-LABEL: depthwise_conv2d_same_padding_dyn
// CHECK: tfl.depthwise_conv_2d
func.func @depthwise_conv2d_same_padding_dyn(%arg0: tensor<?x32x?x8xf32>, %arg1 : tensor<1x1x1x16xf32>, %arg2 : tensor<16xf32>) -> tensor<?x?x?x?xf32> {
  %2 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 3 : i32} : (tensor<?x32x?x8xf32>, tensor<1x1x1x16xf32>, tensor<16xf32>) -> tensor<?x?x?x?xf32>
  func.return %2 : tensor<?x?x?x?xf32>
}
