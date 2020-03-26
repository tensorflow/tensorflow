// Run optimize pass only and check the results.
// RUN: tf-opt %s -tfl-optimize | FileCheck %s --dump-input-on-failure
// Run optimize pass and then canonicalize pass, and make sure some folding is applied.
// RUN: tf-opt %s -tfl-optimize -canonicalize | FileCheck --check-prefix=FOLD %s

// Run legalize pass and then optimize pass, and make sure some fusing is applied.
// RUN: tf-opt %s -tfl-legalize-tf -tfl-optimize | FileCheck --check-prefix=Fusing --dump-input-on-failure %s

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

// CHECK-LABEL: fuseAddIntoFollowingConv2d
func @fuseAddIntoFollowingConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>
  %w = constant dense<1.0> : tensor<16x3x3x3xf32>
  %bias = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.conv_2d"(%0, %w, %bias) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

// CHECK-NEXT: %[[w:.*]] = constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
// CHECK-NEXT: %[[b:.*]] = constant dense<[4.150000e+01, 4.250000e+01, 4.350000e+01, 4.450000e+01, 4.550000e+01, 4.650000e+01, 4.750000e+01, 4.850000e+01, 4.950000e+01, 5.050000e+01, 5.150000e+01, 5.250000e+01, 5.350000e+01, 5.450000e+01, 5.550000e+01, 5.650000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[c:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[c]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: fuseSubIntoFollowingConv2d
func @fuseSubIntoFollowingConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<f32>
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>
  %w = constant dense<1.0> : tensor<16x3x3x3xf32>
  %bias = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.conv_2d"(%0, %w, %bias) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

// CHECK-NEXT: %[[w:.*]] = constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
// CHECK-NEXT: %[[b:.*]] = constant dense<[-3.950000e+01, -3.850000e+01, -3.750000e+01, -3.650000e+01, -3.550000e+01, -3.450000e+01, -3.350000e+01, -3.250000e+01, -3.150000e+01, -3.050000e+01, -2.950000e+01, -2.850000e+01, -2.750000e+01, -2.650000e+01, -2.550000e+01, -2.450000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[c:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[c]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: @fuseAddIntoDepthwiseConv2d
func @fuseAddIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseSubIntoDepthwiseConv2d
func @fuseSubIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<0.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseAddIntoFollowingDepthwiseConv2d
func @fuseAddIntoFollowingDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>

  %w = constant dense<1.0> : tensor<3x3x3x16xf32>
  %bias = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.depthwise_conv_2d"(%0, %w, %bias) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

// CHECK-NEXT: %[[w:.*]] = constant dense<1.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-NEXT: %[[b:.*]] = constant dense<[4.150000e+01, 4.250000e+01, 4.350000e+01, 4.450000e+01, 4.550000e+01, 4.650000e+01, 4.750000e+01, 4.850000e+01, 4.950000e+01, 5.050000e+01, 5.150000e+01, 5.250000e+01, 5.350000e+01, 5.450000e+01, 5.550000e+01, 5.650000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[dc:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[dc]] : tensor<256x30x30x16xf32>
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
func @fuseAddWithRelu6IntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1 : tensor<256x30x30x16xf32>

  // CHECK: %cst = constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
  // CHECK-SAME: fused_activation_function = "RELU6"
}

// CHECK-LABEL: fuseMulIntoConv2dWithQDQs
func @fuseMulIntoConv2dWithQDQs(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x3xf32> {
  %cst = constant dense<1.5> : tensor<3xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %w = constant dense<2.0> : tensor<3x3x3x3xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>) -> tensor<3x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %dq, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<256x30x30x3xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x3xf32>, tensor<3xf32>) -> tensor<256x30x30x3xf32>
  return %1 : tensor<256x30x30x3xf32>

  // CHECK: %[[w:.*]] = constant dense<3.000000e+00> : tensor<3x3x3x3xf32>
  // CHECK: %[[cst:.*]] = constant dense<[1.500000e+00, 3.000000e+00, 4.500000e+00]> : tensor<3xf32>
  // CHECK: %[[q:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}
  // CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
  // CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]], %[[cst]])
  // CHECK: return %[[conv]] : tensor<256x30x30x3xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnected
func @fuseMulIntoFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  return %1 : tensor<4x2xf32>

// CHECK:  %[[CONSTANT:.*]] = constant dense<{{\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK:  %[[CONSTANT0:.*]] = constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) {fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}
// CHECK:  return %[[RES]] : tensor<4x2xf32>
}


// CHECK-LABEL: @fuseAddIntoFollowingFullyConnectedWithQDQs
func @fuseAddIntoFollowingFullyConnectedWithQDQs(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %q = "tfl.quantize"(%cst0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0>>
  %dq = "tfl.dequantize"(%q) : (tensor<2x2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %dq, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  return %1 : tensor<4x2xf32>

// CHECK-NEXT: %[[w:.*]] = constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK-NEXT: %[[b:.*]] = constant dense<[6.500000e+00, 1.250000e+01]> : tensor<2xf32>
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[w]])
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq]], %[[b]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseAddIntoFollowingFullyConnected
func @fuseAddIntoFollowingFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  return %1 : tensor<4x2xf32>

// CHECK-NEXT: %[[w:.*]] = constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK-NEXT: %[[b:.*]] = constant dense<[6.500000e+00, 1.250000e+01]> : tensor<2xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseMulIntoFollowingFullyConnected
func @fuseMulIntoFollowingFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = constant dense<1.5> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  return %1 : tensor<4x2xf32>

// CHECK-NEXT: %[[b:.*]] = constant dense<2.000000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[w:.*]] = constant dense<{{\[}}[1.500000e+00, 3.000000e+00], [4.500000e+00, 6.000000e+00]]> : tensor<2x2xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
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

// CHECK:  %[[CONSTANT:.*]] = constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
// CHECK:  %[[CONSTANT0:.*]] = constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
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

// CHECK:  %[[CONSTANT:.*]] = constant dense<{{\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
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
  // We cannot fuse this tfl.mul into the preceding conv op because %cst2 is not broadcast-compatible to %cst0.
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x112x112x2xf32>, tensor<112x2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>

// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0)
// CHECK:  %1 = "tfl.mul"(%0, %cst_1)
// CHECK:  return %1
}

// CHECK-LABEL: @FuseFullyConnectedAddWithNoBias
func @FuseFullyConnectedAddWithNoBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant unit
  %cst2 = constant dense<2.0> : tensor<40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>

  return %1 : tensor<40x40xf32>

  // CHECK: %cst = constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %cst)
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithExistingBias
func @FuseFullyConnectedAddWithExistingBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<3.0> : tensor<40xf32>
  %cst2 = constant dense<2.0> : tensor<40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>

  return %1 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithNoBiasAndScalarRhs
func @FuseFullyConnectedAddWithNoBiasAndScalarRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant unit
  %cst2 = constant dense<2.0> : tensor<f32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>

  return %1 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithScalarRhs
func @FuseFullyConnectedAddWithScalarRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<3.0> : tensor<40xf32>
  %cst2 = constant dense<2.0> : tensor<f32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>

  return %1 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithUnfusableRhs
func @FuseFullyConnectedAddWithUnfusableRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant unit
  %cst2 = constant dense<2.0> : tensor<40x40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>

  return %1 : tensor<40x40xf32>

  // CHECK: %[[unit:.*]] = constant unit
  // CHECK: %[[filter:.*]] = constant dense<2.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[fc_result:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[unit]])
  // CHECK: %[[add_result:.*]] = tfl.add %[[fc_result]], %[[filter]]
  // CHECK: return %[[add_result]]
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAddConst
// FOLD-LABEL: @FuseFullyConnectedReshapeAddConst
func @FuseFullyConnectedReshapeAddConst(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<3.0> : tensor<40x40xf32>
  %cst2 = constant dense<2.0> : tensor<40xf32>
  %shape1 = constant dense<[1, 40, 40]> : tensor<3xi32>
  %shape2 = constant dense<[40, 40]> : tensor<2xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40x40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<40x40xf32>, tensor<3xi32>) -> tensor<1x40x40xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x40xf32>, tensor<40xf32>) -> tensor<1x40x40xf32>
  %3 = "tfl.reshape"(%2, %shape2) : (tensor<1x40x40xf32>, tensor<2xi32>) -> tensor<40x40xf32>

  return %3 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = constant dense<5.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: %[[rs2:.*]] = "tfl.reshape"(%[[rs1]]
  // CHECK: return %[[rs2]]

  // FOLD: %[[cst:.*]] = constant dense<5.000000e+00> : tensor<40x40xf32>
  // FOLD: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // FOLD: return %[[fc]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfNotBroadcastable
func @NotReorderReshapeAddIfNotBroadcastable(%arg0: tensor<40x10x4xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<2.0> : tensor<40xf32>
  %shape = constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x10x4xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>
  return %2 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = "tfl.add"(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfNotTailingDim
func @NotReorderReshapeAddIfNotTailingDim(%arg0: tensor<40x40x1xf32>) -> tensor<40x40xf32> {
  %cst = constant dense<2.0> : tensor<1x40xf32>
  %shape = constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x40x1xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<1x40xf32>) -> tensor<40x40xf32>
  return %2 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = "tfl.add"(%[[rs1]]
  // CHECK: return %[[rs2]]
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

// CHECK-LABEL: @L2NormalizePattern2
func @L2NormalizePattern2(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %cst_1 = constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.rsqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.mul"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) {fused_activation_function = "NONE"} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern3
func @L2NormalizePattern3(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %cst_1 = constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) {fused_activation_function = "NONE"} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern4
func @L2NormalizePattern4(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %cst_1 = constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.maximum"(%1, %cst_1) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) {fused_activation_function = "NONE"} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern5
func @L2NormalizePattern5(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %cst_1 = constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.maximum"(%1, %cst_1) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %4: tensor<2xf32>
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

// CHECK-LABEL: @InvalidL2NormalizePattern2
// Epsilon in the add must be < 1e-3
func @InvalidL2NormalizePattern2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %cst_1 = constant dense<[1.0e-1]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %4 : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.div"([[INPUT:%.*]], %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @InvalidL2NormalizePattern3
// Axis must be last dimension.
func @InvalidL2NormalizePattern3(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  return %3: tensor<2x2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.div"([[INPUT:%.*]], %2) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  // CHECK: return %[[RES]]
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

// CHECK-LABEL: fuseTileWithBinaryOp
func @fuseTileWithBinaryOp(%arg0: tensor<1x1xf32>) -> tensor<1x2xf32> {
  %cst = constant dense<[[1,2]]> : tensor<1x2xi32>
  %cst1 = constant dense<[[3.0, 4.0]]> : tensor<1x2xf32>
  %0 = "tfl.sqrt"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.tile"(%0, %cst) : (tensor<1x1xf32>, tensor<1x2xi32>) -> tensor<1x2xf32>
  %2 = "tfl.add"(%cst1, %1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>

  // CHECK: %[[cst:.*]] = constant dense<{{\[\[}}3.000000e+00, 4.000000e+00]]> : tensor<1x2xf32>
  // CHECK: %[[SQRT:[0-9].*]] = "tfl.sqrt"
  // CHECK: %[[RES:[0-9].*]] = "tfl.add"(%[[SQRT]], %[[cst]])
}

// CHECK-LABEL: fuseTileWithBinaryOp1
func @fuseTileWithBinaryOp1(%arg0: tensor<1x1xf32>, %arg1: tensor<1x128xf32>) -> tensor<1x128xf32> {
  %cst_0 = constant dense<1.0> : tensor<f32>
  %cst_1 = constant dense<[1, 128]> : tensor<2xi32>
  %0 = "tfl.add"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %1 = "tfl.sqrt"(%0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "tfl.tile"(%1, %cst_1) : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<1x128xf32>
  %3 = "tfl.div"(%2, %arg1) {fused_activation_function = "NONE"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  return %3 : tensor<1x128xf32>

  // CHECK: %[[cst:.*]] = constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[ADD:[0-9].*]] = "tfl.add"(%arg0, %[[cst]]) {fused_activation_function = "NONE"} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  // CHECK: %[[SQRT:[0-9].*]] = "tfl.sqrt"(%[[ADD]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.div"(%[[SQRT]], %arg1) {fused_activation_function = "NONE"} : (tensor<1x1xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: InvalidFuseTileWithBinaryOp
func @InvalidFuseTileWithBinaryOp(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
  %cst = constant dense<[[1,2]]> : tensor<1x2xi32>
  %cst1 = constant dense<[[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]> : tensor<1x6xf32>
  %0 = "tfl.sqrt"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.tile"(%0, %cst) : (tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<2x6xf32>
  %2 = "tfl.add"(%cst1, %1) {fused_activation_function = "NONE"} : (tensor<1x6xf32>, tensor<2x6xf32>) -> tensor<2x6xf32>
  return %2 : tensor<2x6xf32>

  // CHECK: %[[TILE:[0-9].*]] = "tfl.tile"
}

// CHECK-LABEL: InvalidFuseTileAlreadyBroadcastAlongTileDim
func @InvalidFuseTileAlreadyBroadcastAlongTileDim(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x6x8x1xf32> {
  %cst_1 = constant dense<[1, 6, 8, 1]> : tensor<4xi32>
  %cst_2 = constant dense<[1, 1, 1, 46]> : tensor<4xi32>
  %cst_20 = constant dense<4.600000e+01> : tensor<f32>
  %0 = "tfl.tile"(%arg0, %cst_1) : (tensor<1x1x1x1xf32>, tensor<4xi32>) -> tensor<1x6x8x1xf32>
  %1 = "tfl.mul"(%0, %cst_20) {fused_activation_function = "NONE"} : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  return %1 : tensor<1x6x8x1xf32>

  // CHECK: %[[TILE:[0-9].*]] = "tfl.tile"
}

// CHECK-LABEL: FuseHardswish
func @FuseHardswish(%arg0: tensor<1x112x112x16xf32>) -> tensor<1x56x56x16xf32> {
  %cst_0 = constant dense<3.0> : tensor<f32>
  %cst_1 = constant dense<0.166666666> : tensor<f32>
  %w = constant dense<1.0> : tensor<1x3x3x16xf32>
  %b = constant dense<10.0> : tensor<16xf32>
  %2 = "tfl.add"(%arg0, %cst_0) {fused_activation_function = "RELU6"} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
  %3 = "tfl.mul"(%2, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
  %4 = tfl.mul %arg0, %3 {fused_activation_function = "NONE"} : tensor<1x112x112x16xf32>
  %5 = "tfl.depthwise_conv_2d"(%4, %w, %b) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x16xf32>, tensor<1x3x3x16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
  return %5 : tensor<1x56x56x16xf32>

// CHECK: tfl.hard_swish
// CHECK: tfl.depthwise_conv_2d
}

// CHECK-LABEL: squeezeToReshape
func @squeezeToReshape(%arg0: tensor<1x1x2xf32>) -> tensor<2xf32> {
  %0 = "tfl.squeeze"(%arg0) : (tensor<1x1x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>

  // CHECK: [[CONST:.*]] = constant dense<2> : tensor<1xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<1x1x2xf32>, tensor<1xi32>) -> tensor<2xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: expandDimsToReshape
func @expandDimsToReshape(%arg0: tensor<6x6x256xf32>) -> tensor<6x6x256x1xf32> {
  %cst = constant dense<-1> : tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<6x6x256xf32>, tensor<i32>) -> tensor<6x6x256x1xf32>
  return %0 : tensor<6x6x256x1xf32>

  // CHECK: [[CONST:.*]] = constant dense<[6, 6, 256, 1]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<6x6x256xf32>, tensor<4xi32>) -> tensor<6x6x256x1xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: convertTrivialTransposeToReshape
func @convertTrivialTransposeToReshape(%arg0: tensor<6x6x256x1xf32>) -> tensor<1x6x6x256xf32> {
  %cst = constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  return %0 : tensor<1x6x6x256xf32>

  // CHECK: [[CONST:.*]] = constant dense<[1, 6, 6, 256]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: doNotConvertNonTrivialTransposeToReshape
func @doNotConvertNonTrivialTransposeToReshape(%arg0: tensor<6x6x256x1xf32>) -> tensor<1x6x6x256xf32> {
  // Note: The dimension 0 and 1 are swapped, so it's not trivial
  // (elements are not in the same order).
  %cst = constant dense<[3, 1, 0, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  return %0 : tensor<1x6x6x256xf32>

  // CHECK: [[CONST:.*]] = constant dense<[3, 1, 0, 2]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose"(%arg0, %[[CONST:.*]])
  // CHECK: return %[[RESULT]]
}


// CHECK-LABEL: Relu1
func @Relu1(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = constant dense<-1.0> : tensor<f32>
  %cst1 = constant dense<1.0> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.minimum"(%0, %cst1) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

  // CHECK: %[[relu_n1_to_1:[0-9].*]] = "tfl.relu_n1_to_1"
}

// CHECK-LABEL: Relu1_2
func @Relu1_2(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = constant dense<-1.0> : tensor<f32>
  %cst1 = constant dense<1.0> : tensor<f32>
  %0 = "tfl.minimum"(%arg0, %cst1) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %cst) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

  // CHECK: %[[relu_n1_to_1:[0-9].*]] = "tfl.relu_n1_to_1"
}

// CHECK-LABEL: fuse_relu_to_add
func @fuse_relu_to_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "RELU_N1_TO_1"}
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: leaky_relu_fusion
func @leaky_relu_fusion(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = constant dense<0.2> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %alpha) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.leaky_relu"
}

// CHECK-LABEL: leaky_relu_not_fused
// Should not fuse to LeakyRelu, since alpha > 1.
func @leaky_relu_not_fused(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = constant dense<1.2> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %alpha) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.maximum"
}

// CHECK-LABEL: prelu_fusion
func @prelu_fusion(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = constant dense<-0.2> : tensor<3xf32>
  %0 = "tfl.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tfl.relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tfl.mul"(%alpha, %2) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.prelu"
}

// CHECK-LABEL: prelu_not_fused
// Rank of alpha should be one less than input for PReLU, which is not the case.
func @prelu_not_fused(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = constant dense<-0.2> : tensor<f32>
  %0 = "tfl.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tfl.relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tfl.mul"(%alpha, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.relu"
}

// CHECK-LABEL: NotfuseAddIntoConv2d_MultipleUsers
func @NotfuseAddIntoConv2d_MultipleUsers(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> (tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>) {
  %cst = constant dense<1.5> : tensor<16xf32>
  %cst_1 = constant dense<3.5> : tensor<16xf32>
  %cst_0 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %2 = "tfl.add"(%0, %cst_1) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %1, %2 : tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>

  // CHECK: %[[tfl_conv2d:[0-9].*]] = "tfl.conv_2d"
  // CHECK: tfl.add
  // CHECK-NEXT: tfl.add
}

func @FusingaddRelu(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.Relu"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tf.Add"(%3, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "tf.Relu6"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  %6 = "tfl.add"(%5, %3) {fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %7 = "tf.Relu6"(%6) : (tensor<1xf32>) -> tensor<1xf32>
  return %7: tensor<1xf32>

// Fusing-LABEL: FusingaddRelu
// Fusing:  %[[add:[0-9].*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xf32>
// Fusing:  %[[add1:[0-9].*]] = tfl.add %arg0, %[[add]] {fused_activation_function = "RELU"} : tensor<1xf32>
// Fusing:  %[[relu:[0-9].*]] = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
// Fusing:  %[[add2:[0-9].*]] = tfl.add %[[relu]], %[[add1]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  %[[add3:[0-9].*]] = tfl.add %[[add2]], %[[relu]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  return
}

func @FusingbiasAdd(%arg0: tensor<1x10x10x32xf32>, %arg1: tensor<32xf32>) -> tensor<1x10x10x32xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  %1 = "tf.BiasAdd"(%0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  %2 = "tf.Relu6"(%1) : (tensor<1x10x10x32xf32>) -> tensor<1x10x10x32xf32>
  return %2 : tensor<1x10x10x32xf32>

// Fusing-LABEL: FusingbiasAdd
// Fusing:  %[[add:[0-9].*]] = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
// Fusing:  %[[add1:[0-9].*]] = "tfl.add"(%[[add]], %arg1) {fused_activation_function = "RELU6"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
}

func @FusingdivRelu(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.Div"(%arg0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.Relu"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tf.Div"(%3, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "tf.Relu6"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  return %5: tensor<1xf32>

// Fusing-LABEL: FusingdivRelu
// Fusing:  %[[div:[0-9].*]] = tfl.div %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xf32>
// Fusing:  %[[div1:[0-9].*]] = tfl.div %arg0, %[[div]] {fused_activation_function = "RELU"} : tensor<1xf32>
// Fusing:  %[[relu:[0-9].*]] = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
// Fusing:  %[[div2:[0-9].*]] = tfl.div %[[relu]], %[[div1]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  return
}
