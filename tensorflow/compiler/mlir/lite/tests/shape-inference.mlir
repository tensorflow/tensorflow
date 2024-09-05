// RUN: tf-opt -split-input-file -verify-diagnostics --tf-shape-inference %s | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeValidPadding
func.func @testConv2dShapeValidPadding(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x108x76x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceSamePadding
func.func @testConv2dShapeInferenceSamePadding(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x112x80x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceDilation
func.func @testConv2dShapeInferenceDilation(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x112x80x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceStrides
func.func @testConv2dShapeInferenceStrides(%arg0: tensor<1x112x80x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x56x40x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x80x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceUnranked
func.func @testConv2dShapeInferenceUnranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testConv2dShapeInferenceDynamic
func.func @testConv2dShapeInferenceDynamic(%arg0: tensor<1x?x?x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x?x?x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x?x?x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
func.func @testConv2dShapeInvalidRanks(%arg0: tensor<1x112x80xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x?x?x128xf32> {
  // expected-error @+2 {{'tfl.conv_2d' op failed to infer returned types}}
  // expected-error @+1 {{Invalid ranks}}
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x80xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x?x?x128xf32>
  func.return %0 : tensor<1x?x?x128xf32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testUnidirectionalSequenceLstmShapeInference
func.func @testUnidirectionalSequenceLstmShapeInference(%arg0: tensor<600 x 10 x 20 x f32>, %arg1: tensor<? x ? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<40 x f32>, %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<600 x 40 x f32>, %arg19: tensor<600 x 40 x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? x f32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) <{fused_activation_function = "NONE", time_major = false}> : (tensor<600x10x20xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<40xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<600x40xf32>, tensor<600x40xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<600x10x40xf32
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<600 x 10 x 20 x f32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<40xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<600x40xf32>, tensor<600x40xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<? x ? x ? xf32>
  func.return %0 : tensor<? x ? x ? x f32>
}
}

// -----

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: testUnidirectionalSequenceLstmShapeInference
func.func @testUnidirectionalSequenceLstmShapeInference(%arg0: tensor<600 x ? x 20 x f32>, %arg1: tensor<? x ? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<40 x f32>, %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<600 x 40 x f32>, %arg19: tensor<600 x 40 x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? x f32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) <{fused_activation_function = "NONE", time_major = false}> : (tensor<600x?x20xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<40xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<600x40xf32>, tensor<600x40xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<600x?x40xf32
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<600 x ? x 20 x f32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<40xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<600x40xf32>, tensor<600x40xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<? x ? x ? xf32>
  func.return %0 : tensor<? x ? x ? x f32>
}
}

// -----

// CHECK-LABEL: testReshapeShapeInference
module attributes {tf.versions = {producer = 888 : i32}} {
func.func @testReshapeShapeInference(%arg0: tensor<3x4xi32>) -> tensor<*xi32> {
  %cst = arith.constant dense<[1, 6, 2]> : tensor<3xi32>
  // CHECK: "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<1x6x2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
}

// -----

// CHECK-LABEL: testReshapeShapeInferenceUnknownDim
module attributes {tf.versions = {producer = 888 : i32}} {
func.func @testReshapeShapeInferenceUnknownDim(%arg0: tensor<3x4xi32>) -> tensor<*xi32> {
  %cst = arith.constant dense<[1, 6, -1]> : tensor<3xi32>
  // CHECK: "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<1x6x2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
}
