// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range -tfl-quantize="enable-dynamic-range-quantization=true" | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range -tfl-quantize="enable-dynamic-range-quantization=true enable-weight-only-quantization=true" | FileCheck --check-prefix=PerChannelWeightOnly %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-dynamic-range-per-channel-quantization=false" -tfl-quantize="enable-dynamic-range-quantization=true" | FileCheck --check-prefix=PerTensor %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-dynamic-range-per-channel-quantization=false" -tfl-quantize="enable-dynamic-range-quantization=true enable-weight-only-quantization=true" | FileCheck --check-prefix=PerTensorWeightOnly %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-dynamic-range-per-channel-quantization=false" -tfl-quantize="enable-dynamic-range-quantization=true ops-blocklist=tfl.conv_2d" | FileCheck --check-prefix=BLOCK %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1" -tfl-quantize="enable-dynamic-range-quantization=true enable-custom-op-weight-only=CustomTestOp=true" | FileCheck --check-prefix=CustomOpWeightOnly %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1" -tfl-quantize="enable-dynamic-range-quantization=true enable-custom-op-weight-only=CustomTestOp=false" | FileCheck --check-prefix=CustomOpNotWeightOnly %s

// CHECK-LABEL: QuantizeConv2D
// PerTensor-LABEL: QuantizeConv2D
// PerChannelWeightOnly-LABEL: QuantizeConv2D
// PerTensorWeightOnly-LABEL: QuantizeConv2D
// BLOCK-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  func.return %conv : tensor<1x112x112x64xf32>

// CHECK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: dilation_h_factor = 1 : i32
// CHECK: return %[[conv:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: dilation_h_factor = 1 : i32
// PerTensor: return %[[conv:.*]]

// PerChannelWeightOnly: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerChannelWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {
// PerChannelWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {
// PerChannelWeightOnly: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerChannelWeightOnly-NOT: asymmetric_quantize_inputs = true
// PerChannelWeightOnly-SAME: dilation_h_factor = 1 : i32
// PerChannelWeightOnly: return %[[conv:.*]]

// PerTensorWeightOnly: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensorWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensorWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensorWeightOnly: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensorWeightOnly-NOT: asymmetric_quantize_inputs = true
// PerTensorWeightOnly-SAME: dilation_h_factor = 1 : i32
// PerTensorWeightOnly: return %[[conv:.*]]

// BLOCK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// BLOCK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// BLOCK: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// BLOCK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// BLOCK: return %[[conv:.*]]
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// PerTensor-LABEL: QuantizeDepthwiseConv2D
func.func @QuantizeDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<127.0> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  func.return %dconv : tensor<1x112x112x64xf32>

// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: depth_multiplier = 4 : i32
// CHECK: return %[[dconv:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: depth_multiplier = 4 : i32
// PerTensor: return %[[dconv:.*]]
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
// PerChannelWeightOnly-LABEL: QuantizeFullyConnected
// PerTensorWeightOnly-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x512xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %b = arith.constant dense<0.0> : tensor<512xf32>
  %fc = "tfl.fully_connected"(%arg0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<512x12xf32>, tensor<512xf32>) -> tensor<1x112x112x512xf32>
  func.return %fc : tensor<1x112x112x512xf32>

// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) {
// CHECK-NOT: fused_activation_function = "NONE",
// CHECK-SAME: asymmetric_quantize_inputs = true,
// CHECK: return %[[fc:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) {
// PerTensor-NOT: fused_activation_function = "NONE",
// PerTensor-SAME: asymmetric_quantize_inputs = true,
// PerTensor: return %[[fc:.*]]

// PerChannelWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerChannelWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerChannelWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerChannelWeightOnly: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w]], %[[b]]) {
// PerChannelWeightOnly-NOT: fused_activation_function = "NONE",
// PerChannelWeightOnly-SAME: asymmetric_quantize_inputs = true,
// PerChannelWeightOnly: return %[[fc:.*]]

// PerTensorWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensorWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensorWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensorWeightOnly: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensorWeightOnly-NOT: fused_activation_function = "NONE",
// PerTensorWeightOnly-SAME: asymmetric_quantize_inputs = true,
// PerTensorWeightOnly: return %[[fc:.*]]

}

// CHECK-LABEL: QuantizeMatmulWithActConst
// PerTensor-LABEL: QuantizeMatmulWithActConst
func.func @QuantizeMatmulWithActConst(%arg0: tensor<1x3x3x512xf32>) -> tensor<1x3x3x12xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %mm = "tfl.batch_matmul"(%arg0, %w) {adj_x = false, adj_y = false} : (tensor<1x3x3x512xf32>, tensor<512x12xf32>) -> tensor<1x3x3x12xf32>
  func.return %mm : tensor<1x3x3x12xf32>

// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>,
// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[w]]) {adj_x = false, adj_y = false
// CHECK-SAME: , asymmetric_quantize_inputs = true
// CHECK: return %[[mm:.*]]

// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>,
// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[w]]) {adj_x = false, adj_y = false
// PerTensor-SAME: , asymmetric_quantize_inputs = true
// PerTensor: return %[[mm:.*]]
}

// CHECK-LABEL: QuantizeTransposeConvWeightOnly
// PerTensor-LABEL: QuantizeTransposeConvWeightOnly
// PerChannelWeightOnly-LABEL: QuantizeTransposeConvWeightOnly
// PerTensorWeightOnly-LABEL: QuantizeTransposeConvWeightOnly
func.func @QuantizeTransposeConvWeightOnly(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tconv = "tfl.transpose_conv"(%arg1, %w, %arg0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  func.return %tconv : tensor<1x32x42x128xf32>

// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// CHECK: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>) -> tensor<1x32x42x128xf32>
// CHECK: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w]], %arg0, %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: padding = "SAME"
// CHECK: return %[[tconv:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensor: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1x32x42x128xf32>
// PerTensor: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w]], %arg0, %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: padding = "SAME"
// PerTensor: return %[[tconv:.*]]

// PerChannelWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerChannelWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>
// PerChannelWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>) -> tensor<1x32x42x128xf32>
// PerChannelWeightOnly: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w]], %arg0, %[[b]]) {
// PerChannelWeightOnly-NOT: asymmetric_quantize_inputs = true
// PerChannelWeightOnly-SAME: padding = "SAME"
// PerChannelWeightOnly: return %[[tconv:.*]]

// PerTensorWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensorWeightOnly: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensorWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1x32x42x128xf32>
// PerTensorWeightOnly: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w]], %arg0, %[[b]]) {
// PerTensorWeightOnly-NOT: asymmetric_quantize_inputs = true
// PerTensorWeightOnly-SAME: padding = "SAME"
// PerTensorWeightOnly: return %[[tconv:.*]]
}

// CHECK-LABEL: QuantizeGatherWeightOnly
// PerTensor-LABEL: QuantizeGatherWeightOnly
func.func @QuantizeGatherWeightOnly(%arg0: tensor<3xi32>) -> tensor<3x3x3x3xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %emb = "tfl.gather"(%w, %arg0) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<64x3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3x3xf32>
  %emb_s = "quantfork.stats"(%emb) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
  func.return %emb_s : tensor<3x3x3x3xf32>

// CHECK: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// CHECK: %[[emb:.*]] = "tfl.gather"(%[[dq_w]], %arg0)
// CHECK: return %[[emb:.*]]

// PerTensor: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[emb:.*]] = "tfl.gather"(%[[dq_w]], %arg0)
// PerTensor: return %[[emb:.*]]
}

// CHECK-LABEL: QuantizeCustomOp
// CustomOpWeightOnly-LABEL: QuantizeCustomOp
// CustomOpNotWeightOnly-LABEL: QuantizeCustomOp
func.func @QuantizeCustomOp(%arg0: tensor<1x1x1x1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %w = arith.constant dense<127.0> : tensor<1024x1x1x1xf32>
  %custom = "tfl.custom"(%0, %w) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom : tensor<*xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1024x1x1x1xf32>
// CHECK: %[[custom:.*]] = "tfl.custom"(%arg0, %[[w:.*]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">}
// CHECK: return %[[custom:.*]]

// CustomOpWeightOnly: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOpWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
// CustomOpWeightOnly: %[[custom:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">}
// CustomOpWeightOnly: return %[[custom:.*]]

// CustomOpNotWeightOnly: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOpNotWeightOnly: %[[custom:.*]] = "tfl.custom"(%arg0, %[[q_w:.*]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">}
// CustomOpNotWeightOnly: return %[[custom:.*]]
}

// CHECK-LABEL: NotQuantizeConv3D
// PerTensor-LABEL: NotQuantizeConv3D
// PerChannelWeightOnly-LABEL: NotQuantizeConv3D
// PerTensorWeightOnly-LABEL: NotQuantizeConv3D
func.func @NotQuantizeConv3D(%arg0: tensor<1x32x32x32x8xf32>) -> tensor<1x32x32x32x16xf32> {
  %w = arith.constant dense<127.0> : tensor<1x1x1x8x16xf32>
  %b = arith.constant dense<0.0> : tensor<16xf32>
  %conv_3d = "tfl.conv_3d"(%arg0, %w, %b) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x32x8xf32>, tensor<1x1x1x8x16xf32>, tensor<16xf32>) -> tensor<1x32x32x32x16xf32>
  func.return %conv_3d : tensor<1x32x32x32x16xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x1x8x16xf32>
// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[conv_3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[b]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}
// CHECK: return %[[conv_3d:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x1x8x16xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// PerTensor: %[[conv_3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[b]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}
// PerTensor: return %[[conv_3d:.*]]

// PerChannelWeightOnly: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x1x8x16xf32>
// PerChannelWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// PerChannelWeightOnly: %[[conv_3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[b]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}
// PerChannelWeightOnly: return %[[conv_3d:.*]]

// PerTensorWeightOnly: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x1x8x16xf32>
// PerTensorWeightOnly: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// PerTensorWeightOnly: %[[conv_3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[b]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}
// PerTensorWeightOnly: return %[[conv_3d:.*]]
}

// CHECK-LABEL: QuantizeMultiUses
// PerTensor-LABEL: QuantizeMultiUses
// BLOCK-LABEL: QuantizeMultiUses
func.func @QuantizeMultiUses(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<3xi32>) -> (tensor<1x112x112x122xf32>, tensor<3x3x3x3xf32>) {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %emb = "tfl.gather"(%w, %arg1) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<64x3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3x3xf32>
  %bmm = "tfl.batch_matmul"(%conv, %dconv) {adj_x = false, adj_y = true} : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x122xf32>
  func.return %bmm, %emb : tensor<1x112x112x122xf32>, tensor<3x3x3x3xf32>

// CHECK-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK-DAG: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK-DAG: %[[dq_w1:.*]] = "tfl.dequantize"(%[[w1]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// CHECK-DAG: %[[w2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK-DAG: %[[w3:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w3]], %[[b]])
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w2]], %[[b]])
// CHECK: %[[emb:.*]] = "tfl.gather"(%[[dq_w1]], %arg1)
// CHECK: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// CHECK-NOT: , asymmetric_quantize_inputs = true
// CHECK-SAME: }
// CHECK: return %[[bmm:.*]], %[[emb:.*]]

// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w1:.*]] = "tfl.dequantize"(%[[w1]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[w1]], %[[b]])
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w1]], %[[b]])
// PerTensor: %[[emb:.*]] = "tfl.gather"(%[[dq_w1]], %arg1)
// PerTensor: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// PerTensor-NOT: , asymmetric_quantize_inputs = true
// PerTensor-SAME: }
// PerTensor: return %[[bmm:.*]], %[[emb:.*]]

// PerChannelWeightOnly-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerChannelWeightOnly-DAG: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerChannelWeightOnly-DAG: %[[dq_w1:.*]] = "tfl.dequantize"(%[[w1]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerChannelWeightOnly-DAG: %[[w2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// PerChannelWeightOnly-DAG: %[[dq_w2:.*]] = "tfl.dequantize"(%[[w2]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}>>) -> tensor<64x3x3x3xf32>
// PerChannelWeightOnly-DAG: %[[w3:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// PerChannelWeightOnly-DAG: %[[dq_w3:.*]] = "tfl.dequantize"(%[[w3]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// PerChannelWeightOnly: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w3]], %[[b]])
// PerChannelWeightOnly: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w2]], %[[b]])
// PerChannelWeightOnly: %[[emb:.*]] = "tfl.gather"(%[[dq_w1]], %arg1)
// PerChannelWeightOnly: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// PerChannelWeightOnly-NOT: , asymmetric_quantize_inputs = true
// PerChannelWeightOnly-SAME: }
// PerChannelWeightOnly: return %[[bmm:.*]], %[[emb:.*]]

// BLOCK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// BLOCK: %[[w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// BLOCK: %[[dq_w1:.*]] = "tfl.dequantize"(%[[w1]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// BLOCK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w1]], %[[b]])
// BLOCK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w1]], %[[b]])
// BLOCK: %[[emb:.*]] = "tfl.gather"(%[[dq_w1]], %arg1)
// BLOCK: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// BLOCK: return %[[bmm:.*]], %[[emb:.*]]
}
