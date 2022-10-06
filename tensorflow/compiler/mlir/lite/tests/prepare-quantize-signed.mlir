// RUN: tf-opt %s -tfl-prepare-quantize="quantize-signed=true" | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize="quantize-signed=true disable-per-channel=true" | FileCheck --check-prefix=PerTensor %s

// CHECK-LABEL: uint8_to_int8
func.func @uint8_to_int8(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x2xf32>
  func.return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>} : (tensor<2x2xf32>)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_per_axis
func.func @uint8_to_int8_per_axis(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>) -> tensor<2x2xf32>
  func.return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+00,1.000000e+00:-128}>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%0)
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_narrow_range
func.func @uint8_to_int8_narrow_range(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>) -> tensor<2x2xf32>
  func.return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00:127>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: prepareStatistics
func.func @prepareStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quantfork.stats"(%0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  func.return %1 : tensor<8x4x3xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<8x4x3x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>, volatile}
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]])
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[dq1]]) {qtype = tensor<8x4x3x!quant.uniform<i8:f32:2, {0.0078431372549019607:-1,0.062745098039215685:-1,0.0039215686274509803:-1}>>, volatile}
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
// CHECK: return %[[dq2]]
}

// CHECK-LABEL: prepareStatisticsNudge
func.func @prepareStatisticsNudge(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[0.1, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quantfork.stats"(%0) {
    layerStats = dense<[0.1, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, -1.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  func.return %1 : tensor<8x4x3xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<8x4x3x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>, volatile}
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]])
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[dq1]]) {qtype = tensor<8x4x3x!quant.uniform<i8:f32:2, {0.0078431372549019607:-1,0.031372549019607843:127,0.0039215686274509803:-1}>>, volatile}
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
// CHECK: return %[[dq2]]
}

// CHECK-LABEL: preparePrelu
func.func @preparePrelu(%arg0: tensor<1x10x10x3xf32>) -> tensor<1x10x10x3xf32> {
  %cst = "arith.constant"() {value = dense<[[[1.66394591, 3.61694336, 2.0382936]]]> : tensor<1x1x3xf32>} : () -> tensor<1x1x3xf32>
  %prelu = "tfl.prelu"(%arg0, %cst) : (tensor<1x10x10x3xf32>, tensor<1x1x3xf32>) -> tensor<1x10x10x3xf32>
  func.return %prelu : tensor<1x10x10x3xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<[{{\[}}[1.66394591, 3.61694336, 2.0382936]]]> : tensor<1x1x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<1x1x3x!quant.uniform<i8<-127:127>:f32, 0.028479868971456691>>, volatile} : (tensor<1x1x3xf32>) -> tensor<1x1x3x!quant.uniform<i8<-127:127>:f32, 0.028479868971456691>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]]) : (tensor<1x1x3x!quant.uniform<i8<-127:127>:f32, 0.028479868971456691>>) -> tensor<1x1x3xf32>
// CHECK: %[[p:.*]] = "tfl.prelu"(%arg0, %[[dq]]) : (tensor<1x10x10x3xf32>, tensor<1x1x3xf32>) -> tensor<1x10x10x3xf32>
// CHECK: return %[[p]] : tensor<1x10x10x3xf32>
}

// CHECK-LABEL: prepareAdd
func.func @prepareAdd(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<[[0.0, 1.0], [2.0, 255.0]]> : tensor<2x2xf32>
  %add = "tfl.add"(%arg0, %cst) {fused_activation_function="NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %add : tensor<2x2xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<[{{\[}}0.000000e+00, 1.000000e+00], [2.000000e+00, 2.550000e+02]]>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:-128>>, volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[add:.*]] = tfl.add %arg0, %[[dq]]
// CHECK: return %[[add]]
}

// CHECK-LABEL: prepareConv2DSplat
// PerTensor-LABEL: prepareConv2DSplat
func.func @prepareConv2DSplat(%arg0: tensor<1x5x5x3xf32>) -> tensor<1x5x5x3xf32> {
  %w = arith.constant dense<127.0> : tensor<3x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<3xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x5x5x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  func.return %conv : tensor<1x5x5x3xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0
// CHECK-SAME:  {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x3xf32>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, volatile}
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]
}

// CHECK-LABEL: prepareConv2D
// PerTensor-LABEL: prepareConv2D
func.func @prepareConv2D(%arg0: tensor<1x5x5x1xf32>) -> tensor<1x5x5x3xf32> {
  %w = arith.constant dense<[[[[0.0]]], [[[127.0]]], [[[-127.0]]]]> : tensor<3x1x1x1xf32>
  %b = arith.constant dense<0.0> : tensor<3xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x5x5x1xf32>, tensor<3x1x1x1xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  func.return %conv : tensor<1x5x5x3xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<[{{\[\[\[}}0.000000e+00]]], [{{\[\[}}1.270000e+02]]], [{{\[\[}}-1.270000e+02]]]]>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x1x1x1x!quant.uniform<i8<-127:127>:f32:0,
// CHECK-SAME: {3.9370078740157481E-9,1.000000e+00,1.000000e+00}>>, volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = arith.constant dense<[{{\[\[\[}}0.000000e+00]]], [{{\[\[}}1.270000e+02]]], [{{\[\[}}-1.270000e+02]]]]>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<3x1x1x1x!quant.uniform<i8<-127:127>:f32,
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]]
}

// CHECK-LABEL: prepareDepthwiseConv2D
// PerTensor-LABEL: prepareDepthwiseConv2D
func.func @prepareDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  %w = arith.constant dense<127.0> : tensor<32x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<32xf32>
  %dc = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %dc : tensor<1x112x112x32xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<32x3x3x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32:3
// CHECK-SAME:  {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<32x3x3x3xf32>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32,
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// PerTensor: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq]]
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  %w = arith.constant dense<127.0> : tensor<32x12xf32>
  %b = arith.constant dense<0.0> : tensor<32xf32>
  %fc = "tfl.fully_connected"(%arg0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %fc : tensor<1x112x112x32xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<32x12xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%cst) {qtype = tensor<32x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%0) : (tensor<32x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<32x12xf32>
// CHECK: "tfl.fully_connected"(%arg0, %[[dq]]

// PerTensor: %[[cst:.*]] = arith.constant dense<1.270000e+02> : tensor<32x12xf32>
// PerTensor: %[[q:.*]] = "tfl.quantize"(%cst) {qtype = tensor<32x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, volatile}
// PerTensor: %[[dq:.*]] = "tfl.dequantize"(%0) : (tensor<32x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<32x12xf32>
// PerTensor: "tfl.fully_connected"(%arg0, %[[dq]]
}

// CHECK-LABEL: QuantizeTransposeConv
// PerTensor-LABEL: QuantizeTransposeConv
func.func @QuantizeTransposeConv(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tc = "tfl.transpose_conv"(%arg1, %w, %arg0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  func.return %tc : tensor<1x32x42x128xf32>

// CHECK: %[[CST:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CST]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>, volatile}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>) -> tensor<1x32x42x128xf32>
// CHECK: "tfl.transpose_conv"(%arg1, %[[DEQUANTIZE]], %arg0,

// PerTensor: %[[CST:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// PerTensor: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CST]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, volatile}
// PerTensor: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1x32x42x128xf32>
// PerTensor: "tfl.transpose_conv"(%arg1, %[[DEQUANTIZE]], %arg0,
}

// CHECK-LABEL: bias_adjust_pertensor
func.func @bias_adjust_pertensor(%arg0: tensor<1x2xf32>) -> (tensor<1x2xf32>) {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.28e-5, 1.27e-5]> : tensor<2xf32>
  } : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %w = arith.constant dense<[[0.0, 1.0], [1.0, 2.0]]> : tensor<2x2xf32>
  %b = arith.constant dense<[0.0, 2.1473647e6]> : tensor<2xf32>
  %fc = "tfl.fully_connected"(%0, %w, %b) {
    fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"
  } : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  func.return %fc : tensor<1x2xf32>
// CHECK-DAG: %[[weight:.*]] = arith.constant dense<{{\[\[}}0.000000e+00, 1.000000e+00]
// CHECK-DAG: %[[bias:.*]] = arith.constant dense<[0.000000e+00, 2147364.75]>
// CHECK-DAG: %[[b_q:.*]] = "tfl.quantize"(%[[bias]]){{.*}}quant.uniform<i32:f32, 0.0019998892694710656>>
// CHECK-DAG: %[[w_q:.*]] = "tfl.quantize"(%[[weight]]){{.*}}quant.uniform<i8<-127:127>:f32, 19998.892343977564>>
// CHECK-DAG: %[[b_dq:.*]] = "tfl.dequantize"(%[[b_q]])
// CHECK-DAG: %[[w_dq:.*]] = "tfl.dequantize"(%[[w_q]])
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%[[input:.*]], %[[w_dq]], %[[b_dq]])
// CHECK: return %[[fc]]
}

// CHECK-LABEL: bias_adjust_perchannel
func.func @bias_adjust_perchannel(%arg0: tensor<1x5x5x2xf32>, %arg1: tensor<4xi32>) -> (tensor<1x5x5x3xf32>) {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.28e-5, 1.27e-5]> : tensor<2xf32>
  } : (tensor<1x5x5x2xf32>) -> tensor<1x5x5x2xf32>
  %w = arith.constant dense<[[[[-1.0, 1.0]]], [[[1.0, 2.0]]], [[[-2.0, 1.0]]]]> : tensor<3x1x1x2xf32>
  %b = arith.constant dense<[1.0e-2, 2.1473647e1, -2.1473647e2]> : tensor<3xf32>
  %transpose_conv = "tfl.transpose_conv"(%arg1, %w, %0, %b) {
    padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32
  } : (tensor<4xi32>, tensor<3x1x1x2xf32>, tensor<1x5x5x2xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  func.return %transpose_conv : tensor<1x5x5x3xf32>
// CHECK: %[[bias:.*]] = arith.constant dense<[0.00999999977, 21.4736462, -214.736465]>
// CHECK: %[[b_q:.*]] = "tfl.quantize"(%[[bias]])
// CHECK-SAME: {7.8740158861230386E-10,1.9998891450408216E-8,1.9998891805679583E-7}
// CHECK: %[[b_dq:.*]] = "tfl.dequantize"(%[[b_q]])
// CHECK: %[[weight:.*]] = arith.constant dense<{{\[\[\[\[}}-1.000000e+00, 1.000000e+00]]]
// CHECK: %[[w_q:.*]] = "tfl.quantize"(%[[weight]])
// CHECK-SAME: {0.0078740157480314959,0.19998891099675145,1.9998891454946508}
// CHECK: %[[w_dq:.*]] = "tfl.dequantize"(%[[w_q]])
// CHECK: %[[conv:.*]] = "tfl.transpose_conv"(%arg1, %[[w_dq]], %[[input:.*]], %[[b_dq]])
// CHECK: return %6 : tensor<1x5x5x3xf32>
}

// CHECK-LABEL: bias_adjust_duplicate_filter
func.func @bias_adjust_duplicate_filter(%arg0: tensor<1x5x5x2xf32>) -> (tensor<1x5x5x3xf32>, tensor<1x5x5x3xf32>) {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.28e-5, 1.27e-5]> : tensor<2xf32>
  } : (tensor<1x5x5x2xf32>) -> tensor<1x5x5x2xf32>
  %w = arith.constant dense<[[[[-1.0, 1.0]]], [[[1.0, 2.0]]], [[[-2.0, 1.0]]]]> : tensor<3x1x1x2xf32>
  %b = arith.constant dense<0.0> : tensor<3xf32>
  %b2 = arith.constant dense<[1.0e-2, 2.1473647e1, -2.1473647e2]> : tensor<3xf32>
  %conv = "tfl.conv_2d"(%0, %w, %b) {
    dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU",
    padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32
  } : (tensor<1x5x5x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  %conv2 = "tfl.conv_2d"(%0, %w, %b2) {
    dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU",
    padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32
  } : (tensor<1x5x5x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<1x5x5x3xf32>
  func.return %conv, %conv2 : tensor<1x5x5x3xf32>, tensor<1x5x5x3xf32>
// CHECK-DAG: %[[bias:.*]] = arith.constant dense<[0.00999999977, 21.4736462, -214.736465]>
// CHECK-DAG: %[[w1:.*]] = arith.constant dense<{{\[\[\[\[}}-1.000000e+00, 1.000000e+00]]]
// CHECK-DAG: %[[b_q:.*]] = "tfl.quantize"(%[[bias]]){{.*}}{7.8740158861230386E-10,1.9998891450408216E-8,1.9998891805679583E-7}
// CHECK-DAG: %[[b_dq:.*]] = "tfl.dequantize"(%[[b_q]])
// CHECK-DAG: %[[w1_q:.*]] = "tfl.quantize"(%[[w1]]){{.*}}{0.0078740157480314959,0.015748031496062992,0.015748031496062992}
// CHECK-DAG: %[[w1_dq:.*]] = "tfl.dequantize"(%[[w1_q]])
// Weight with adjusted scales
// CHECK-DAG: %[[w2:.*]] = arith.constant dense<{{\[\[\[\[}}-1.000000e+00, 1.000000e+00]]]
// CHECK-DAG: %[[w2_q:.*]] = "tfl.quantize"(%[[w2]]){{.*}}{0.0078740157480314959,0.19998891099675145,1.9998891454946508}
// CHECK-DAG: %[[w2_dq:.*]] = "tfl.dequantize"(%[[w2_q]])
// Bias with adjusted scales

// CHECK: %[[conv_normal:.*]] = "tfl.conv_2d"(%[[input:.*]], %[[w1_dq]], %[[bias_normal:.*]])
// CHECK: %[[conv_adjusted:.*]] = "tfl.conv_2d"(%[[input:.*]], %[[w2_dq]], %[[b_dq]])
// CHECK: return %[[conv_normal]], %[[conv_adjusted]]
}

// CHECK-LABEL: bias_adjust_pass_immutable
func.func @bias_adjust_pass_immutable(%arg0: tensor<1x2xf32>) -> (tensor<1x2xf32>) {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.28e-5, 1.27e-5]> : tensor<2xf32>
  } : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %w = arith.constant dense<[[0.0, 1.0], [1.0, 2.0]]> : tensor<2x2xf32>
  %w_q = "quantfork.stats"(%w) {
    layerStats = dense<[0.0, 2.0]> : tensor<2xf32>
  } : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %b = arith.constant dense<[0.0, 2.1473647e3]> : tensor<2xf32>
  %fc = "tfl.fully_connected"(%0, %w_q, %b) {
    fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"
  } : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  func.return %fc : tensor<1x2xf32>
// CHECK: %[[weight:.*]] = arith.constant dense<{{\[\[}}0.000000e+00, 1.000000e+00]
// CHECK: %[[w_q:.*]] = "tfl.quantize"(%[[weight]])
// CHECK-SAME: quant.uniform<i8:f32, 0.0078431372549019607:-128>
}
