// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range -tfl-quantize --tfl-enable-dynamic-range-quantization | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range -tfl-quantize --tfl-enable-dynamic-range-quantization --tfl-enable-dynamic-range-per-channel-quantization=false | FileCheck --check-prefix=PerTensor %s

// CHECK-LABEL: QuantizeConv2D
// PerTensor-LABEL: QuantizeConv2D
func @QuantizeConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %conv : tensor<1x112x112x64xf32>

// CHECK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]])
// CHECK: return %[[conv:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]])
// PerTensor: return %[[conv:.*]]
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// PerTensor-LABEL: QuantizeDepthwiseConv2D
func @QuantizeDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<127.0> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %dconv : tensor<1x112x112x64xf32>

// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]])
// CHECK: return %[[dconv:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]])
// PerTensor: return %[[dconv:.*]]
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x512xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %b = arith.constant dense<0.0> : tensor<512xf32>
  %fc = "tfl.fully_connected"(%arg0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<512x12xf32>, tensor<512xf32>) -> tensor<1x112x112x512xf32>
  return %fc : tensor<1x112x112x512xf32>

// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]])
// CHECK: return %[[fc:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]])
// PerTensor: return %[[fc:.*]]
}

// CHECK-LABEL: NotQuantizeTransposeConv
// PerTensor-LABEL: NotQuantizeTransposeConv
func @NotQuantizeTransposeConv(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tconv = "tfl.transpose_conv"(%arg1, %w, %arg0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  return %tconv : tensor<1x32x42x128xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// CHECK: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[w]], %arg0, %[[b]])
// CHECK: return %[[tconv:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensor: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[w]], %arg0, %[[b]])
// PerTensor: return %[[tconv:.*]]
}

// CHECK-LABEL: QuantizeMultiUses
// PerTensor-LABEL: QuantizeMultiUses
func @QuantizeMultiUses(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x122xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %bmm = "tfl.batch_matmul"(%conv, %dconv) {adj_x = false, adj_y = true} : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x122xf32>
  return %bmm : tensor<1x112x112x122xf32>

// CHECK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[w2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w2]], %[[b]])
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w1]], %[[b]])
// CHECK: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]])
// CHECK: return %[[bmm:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[w2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w2]], %[[b]])
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w1]], %[[b]])
// PerTensor: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]])
// PerTensor: return %[[bmm:.*]]
}
