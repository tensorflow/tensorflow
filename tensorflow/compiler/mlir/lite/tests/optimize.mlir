// RUN: tf-opt %s -tfl-optimize | FileCheck %s

// CHECK-LABEL: fusedConv2dRelu
func @fusedConv2dRelu(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fusedDepthwiseConv2dRelu6
func @fusedDepthwiseConv2dRelu6(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu6"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fusedConv2dTanh
func @fusedConv2dTanh(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.tanh"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "TANH", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fuseAddIntoConv2d
func @fuseAddIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseSubIntoConv2d
func @fuseSubIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: @fuseAddIntoDepthwiseConv2d
func @fuseAddIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseSubIntoDepthwiseConv2d
func @fuseSubIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseAddWithRelu6IntoConv2d
func @fuseAddWithRelu6IntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
  // CHECK-SAME: fused_activation_function = "RELU6"
}

// CHECK-LABEL: @fuseAddWithRelu6IntoDepthwiseConv2d
func @fuseAddWithRelu6IntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
  // CHECK-SAME: fused_activation_function = "RELU6"
}

// CHECK-LABEL: intermOpUsedTwice
func @intermOpUsedTwice(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> (tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>) {
  %cst = constant dense<1.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0, %1 : tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>

  // CHECK:  %cst = constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00,
  // CHECK:  %cst_0 = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00,
  // CHECK:  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}
  // CHECK:  %1 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}
  // CHECK:  return %0, %1

}

// CHECK-LABEL: @fuseMulIntoFullyConnected
func @fuseMulIntoFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  return %1 : tensor<4x2xf32>

// CHECK:  %[[CONSTANT:.*]] = "tf.Const"{{.*}} dense<{{\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK:  %[[CONSTANT0:.*]] = "tf.Const"{{.*}} dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) {fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}
// CHECK:  return %[[RES]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnectedBroadcast
func @fuseMulIntoFullyConnectedBroadcast(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32> {
  %cst0 = constant dense<[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  // %cst2 isn't broadcast-compatible to %cst0, but tf.Mul is able to fold them.
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>

// CHECK:  %[[CONSTANT:.*]] = "tf.Const"{{.*}} dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
// CHECK:  %[[CONSTANT0:.*]] = "tf.Const"{{.*}} dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) {fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}
// CHECK:  return %[[RES]] : tensor<1x2xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnectedNoBias
func @fuseMulIntoFullyConnectedNoBias(%arg0: tensor<4x2xf32>, %arg1: none) -> tensor<4x2xf32> {
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %arg1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  return %1 : tensor<4x2xf32>

// CHECK:  %[[CONSTANT:.*]] = "tf.Const"{{.*}} dense<{{\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %arg1) {fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
// CHECK:  return %[[RES]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseMulIntoDepthwiseConv2d
func @fuseMulIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>

// CHECK:  %cst = constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00], [5.000000e+00, 1.200000e+01]], {{\[\[}}7.000000e+00, 1.600000e+01], [9.000000e+00, 2.000000e+01], [1.100000e+01, 2.400000e+01]], {{\[\[}}1.300000e+01, 2.800000e+01], [1.500000e+01, 3.200000e+01], [1.700000e+01, 3.600000e+01]]]]> : tensor<1x3x3x2xf32>
// CHECK:  %cst_0 = constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
// CHECK:  return %0
}

// CHECK-LABEL: @notFuseMulIntoDepthwiseConv2d
func @notFuseMulIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %cst2 = constant dense<3.0> : tensor<112x2xf32>

  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // We cannot fuse this tfl.mul into the preceding conv op becuase %cst2 is not broadcast-compatible to %cst0.
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x112x112x2xf32>, tensor<112x2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>

// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0)
// CHECK:  %1 = "tfl.mul"(%0, %cst_1)
// CHECK:  return %1
}

// CHECK-LABEL: @FuseFullyConnectedAddUnit
func @FuseFullyConnectedAddUnit(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant unit
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  %cst2 = constant dense<2.0> : tensor<40x40xf32>

  %2 = "tfl.fully_connected" (%0, %1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %3 = "tfl.add"(%2, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>

  return %3 : tensor<40x40xf32>

  // CHECK: %cst = constant dense<2.000000e+00> : tensor<40x40xf32>
  // CHECK: %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK: %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK: %2 = "tfl.fully_connected"(%0, %1, %cst)
  // CHECK: return %2
}

// CHECK-LABEL: @FuseFullyConnectedAddConst
func @FuseFullyConnectedAddConst(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<3.0> : tensor<40x40xf32>
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %cst2 = constant dense<2.0> : tensor<40x40xf32>

  %2 = "tfl.fully_connected" (%0, %1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40x40xf32>) -> (tensor<40x40xf32>)
  %3 = "tfl.add"(%2, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>

  return %3 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = constant dense<5.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[cst_0:.*]] = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK: %[[cst_1:.*1]] = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%[[cst_0]], %[[cst_1]], %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedRelu
func @FuseFullyConnectedRelu(%arg0: tensor<1x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "tfl.fully_connected" (%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x256xf32>, tensor<128x256xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "tfl.relu"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  return %1 : tensor<1x128xf32>

  // CHECK: %[[RES:[0-9].*]] = "tfl.fully_connected"
  // CHECK-SAME: fused_activation_function = "RELU"
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @HardSwishPattern
func @HardSwishPattern(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %three = constant dense<3.> : tensor<f32>
  %six = constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.add"(%arg0, %three)  {fused_activation_function = "RELU6"}  : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.mul"(%1, %six)  {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  return %2: tensor<1xf32>
  // CHECK: %0 = "tfl.hard_swish"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
}

// CHECK-LABEL: @HardSwishPatternFail
func @HardSwishPatternFail(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %three = constant dense<4.> : tensor<f32>
  %six = constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.sub"(%arg0, %three)  {fused_activation_function = "RELU6"}  : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.mul"(%1, %six)  {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  return %2: tensor<1xf32>
  // CHECK: %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
}

// CHECK-LABEL: @L2NormalizePattern
func @L2NormalizePattern(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.rsqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.mul"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %3: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) {fused_activation_function = "NONE"} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern1
func @L2NormalizePattern1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %3: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) {fused_activation_function = "NONE"} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @InvalidL2NormalizePattern
// Div and square ops must take the same argument to be eligible.
func @InvalidL2NormalizePattern(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg1, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %3: tensor<2xf32>
  // CHECK: %3 = "tfl.div"([[INPUT:%.*]], %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: return %3
}

// CHECK-LABEL: @fuseDivIntoConv2d
func @fuseDivIntoConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]> : tensor<2x2x2x2xf32>
  %cst1 = constant dense<1.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %[[cst:.*]] = constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], {{\[\[}}5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]], {{\[\[\[}}4.500000e+00, 5.000000e+00], [5.500000e+00, 6.000000e+00]], {{\[\[}}6.500000e+00, 7.000000e+00], [7.500000e+00, 8.000000e+00]]]]> : tensor<2x2x2x2xf32>
  // CHECK: %[[cst:.*]] = constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseDivIntoDepthwiseConv2d
func @fuseDivIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]> : tensor<2x2x2x2xf32>
  %cst1 = constant dense<1.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %[[cst:.*]] = constant dense<{{\[\[\[\[}}1.000000e+00, 1.000000e+00], [3.000000e+00, 2.000000e+00]], {{\[\[}}5.000000e+00, 3.000000e+00], [7.000000e+00, 4.000000e+00]]], {{\[\[\[}}9.000000e+00, 5.000000e+00], [1.100000e+01, 6.000000e+00]], {{\[\[}}1.300000e+01, 7.000000e+00], [1.500000e+01, 8.000000e+00]]]]> : tensor<2x2x2x2xf32>
  // CHECK: %[[cst:.*]] = constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0) {depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseDivIntoConv2d_Scalar
func @fuseDivIntoConv2d_Scalar(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %cst1 = constant dense<1.0> : tensor<2xf32>
  %cst2 = constant dense<2.0> : tensor<f32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<f32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %[[CST1:.*]] = constant dense<{{\[\[\[\[}}5.000000e-01, 1.000000e+00], [1.500000e+00, 2.000000e+00]], {{\[\[}}2.500000e+00, 3.000000e+00], [3.500000e+00, 4.000000e+00]]]]> : tensor<1x2x2x2xf32>
  // CHECK: %[[CST2:.*]] = constant dense<5.000000e-01> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %[[CST1]], %[[CST2]]) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseMulIntoConv2d_Scalar
func @fuseMulIntoConv2d_Scalar(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %cst1 = constant dense<1.0> : tensor<2xf32>
  %cst2 = constant dense<2.0> : tensor<f32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<f32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %[[CST1:.*]] = constant dense<{{\[\[\[\[}}2.000000e+00, 4.000000e+00], [6.000000e+00, 8.000000e+00]], {{\[\[}}1.000000e+01, 1.200000e+01], [1.400000e+01, 1.600000e+01]]]]> : tensor<1x2x2x2xf32>
  // CHECK: %[[CST2:.*]] = constant dense<2.000000e+00> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %[[CST1]], %[[CST2]]) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}
