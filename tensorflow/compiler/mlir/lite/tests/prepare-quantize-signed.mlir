// RUN: tf-opt %s -tfl-prepare-quantize -tfl-test-quantize-signed | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize -tfl-test-quantize-signed -tfl-disable-per-channel | FileCheck --check-prefix=PerTensor %s

// CHECK-LABEL: uint8_to_int8
func @uint8_to_int8(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>} : (tensor<2x2xf32>)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_per_axis
func @uint8_to_int8_per_axis(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+00,1.000000e+00:-128}>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%0)
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_narrow_range
func @uint8_to_int8_narrow_range(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00:127>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: prepareStatistics
func @prepareStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.stats"(%arg0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quant.stats"(%0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<8x4x3x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>}
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]])
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[dq1]]) {qtype = tensor<8x4x3x!quant.uniform<i8:f32:2, {0.0078431372549019607:-1,0.062745098039215685:-1,0.0039215686274509803:-1}>>}
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
// CHECK: return %[[dq2]]
}

// CHECK-LABEL: prepareAdd
func @prepareAdd(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<[[0.0, 1.0], [2.0, 255.0]]> : tensor<2x2xf32>
  %add = "tfl.add"(%arg0, %cst) {fused_activation_function="NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %add : tensor<2x2xf32>

// CHECK: %[[cst:.*]] = constant dense<[{{\[}}0.000000e+00, 1.000000e+00], [2.000000e+00, 2.550000e+02]]>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:-128>>}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[add:.*]] = tfl.add %arg0, %[[dq]]
// CHECK: return %[[add]]
}

// CHECK-LABEL: prepareConv2DSplat
func @prepareConv2DSplat(%arg0: tensor<1x5x5x3xf32>) -> tensor<1x5x5x3xf32> {
  %w = constant dense<127.0> : tensor<3x3x3x3xf32>
  %b = constant dense<0.0> : tensor<3xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x5x5x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  return %conv : tensor<1x5x5x3xf32>

// CHECK: %[[cst:.*]] = constant dense<1.270000e+02> : tensor<3x3x3x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0
// CHECK-SAME:  {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = constant dense<1.270000e+02> : tensor<3x3x3x3xf32>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]
}

// CHECK-LABEL: prepareConv2D
func @prepareConv2D(%arg0: tensor<1x5x5x1xf32>) -> tensor<1x5x5x3xf32> {
  %w = constant dense<[[[[0.0]]], [[[127.0]]], [[[-127.0]]]]> : tensor<3x1x1x1xf32>
  %b = constant dense<0.0> : tensor<3xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x5x5x1xf32>, tensor<3x1x1x1xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  return %conv : tensor<1x5x5x3xf32>

// CHECK: %[[cst:.*]] = constant dense<[{{\[\[\[}}0.000000e+00]]], [{{\[\[}}1.270000e+02]]], [{{\[\[}}-1.270000e+02]]]]>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x1x1x1x!quant.uniform<i8<-127:127>:f32:0,
// CHECK-SAME: {0.0078740157480314959,1.000000e+00,1.000000e+00}>>}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = constant dense<[{{\[\[\[}}0.000000e+00]]], [{{\[\[}}1.270000e+02]]], [{{\[\[}}-1.270000e+02]]]]>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x1x1x1x!quant.uniform<i8<-127:127>:f32,
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]
}

// CHECK-LABEL: prepareDepthwiseConv2D
func @prepareDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  %w = constant dense<127.0> : tensor<32x3x3x3xf32>
  %b = constant dense<0.0> : tensor<32xf32>
  %dc = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  return %dc : tensor<1x112x112x32xf32>

// CHECK: %[[cst:.*]] = constant dense<1.270000e+02> : tensor<32x3x3x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32:3
// CHECK-SAME:  {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = constant dense<1.270000e+02> : tensor<32x3x3x3xf32>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32,
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq]]
}
