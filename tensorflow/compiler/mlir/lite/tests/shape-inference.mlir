// RUN: tf-opt -split-input-file -verify-diagnostics --tf-shape-inference %s | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeValidPadding
func @testConv2dShapeValidPadding(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x108x76x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceSamePadding
func @testConv2dShapeInferenceSamePadding(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x112x80x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceDilation
func @testConv2dShapeInferenceDilation(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x112x80x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceStrides
func @testConv2dShapeInferenceStrides(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x56x40x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceUnranked
func @testConv2dShapeInferenceUnranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceDynamic
func @testConv2dShapeInferenceDynamic(%arg0: tensor<1x?x?x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x?x?x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x?x?x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
func @testConv2dShapeInvalidRanks(%arg0: tensor<1x112x80xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // expected-error @+1 {{Invalid ranks}}
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  return %0 : tensor<1x?x?x128xf32>
}
}

