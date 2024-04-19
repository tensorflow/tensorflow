// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-dynamic-range-per-channel-quantization=false" | FileCheck --check-prefix=PerTensor %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-float16-quantization" | FileCheck --check-prefix=Float16 %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1-3,CustomTestOp3=3" | FileCheck --check-prefix=CustomOp %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="min-elements-for-weights=4000 enable-custom-op-quantization=CustomTestOp=1-3,CustomTestOp3=3" | FileCheck --check-prefix=MinElement %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="min-elements-for-weights=19" | FileCheck --check-prefix=LSTMOpQuantized %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="min-elements-for-weights=21" | FileCheck --check-prefix=LSTMOpNotQuantized %s

// CHECK-LABEL: QuantizeConv2D
// PerTensor-LABEL: QuantizeConv2D
// MinElement-LABEL: QuantizeConv2D
// Float16-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %conv_s = "quantfork.stats"(%conv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  func.return %conv_s : tensor<1x112x112x64xf32>

// CHECK-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: dilation_h_factor = 1 : i32
// CHECK: return %[[conv:.*]]

// PerTensor-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: dilation_h_factor = 1 : i32
// PerTensor: return %[[conv:.*]]

// MinElement-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// MinElement-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// MinElement: %[[conv:.*]]= "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// MinElement: return %[[conv:.*]]

// Float16-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16-DAG: %[[b:.*]] = arith.constant dense<-1.237300e+00> : tensor<64xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[conv:.*]]= "tfl.conv_2d"(%arg0, %[[dq_w]], %[[dq_b]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// Float16: return %[[conv:.*]]
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// PerTensor-LABEL: QuantizeDepthwiseConv2D
// MinElement-LABEL: QuantizeDepthwiseConv2D
// Float16-LABEL: QuantizeDepthwiseConv2D
func.func @QuantizeDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<127.0> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dconv_s = "quantfork.stats"(%dconv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  func.return %dconv_s : tensor<1x112x112x64xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]])
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: depth_multiplier = 4 : i32
// CHECK: return %[[dconv:.*]]

// PerTensor-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: depth_multiplier = 4 : i32
// PerTensor: return %[[dconv:.*]]

// MinElement: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// MinElement: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// MinElement: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// MinElement: return %[[dconv:.*]]

// Float16-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[dq_b]]) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// Float16: return %[[dconv:.*]]
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x512xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %b = arith.constant dense<0.0> : tensor<512xf32>
  %fc = "tfl.fully_connected"(%0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<512x12xf32>, tensor<512xf32>) -> tensor<1x112x112x512xf32>
  %fc_s = "quantfork.stats"(%fc) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
  func.return %fc : tensor<1x112x112x512xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// CHECK-DAG: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-DAG: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: fused_activation_function = "NONE"
// CHECK-SAME: asymmetric_quantize_inputs = true
// CHECK: return %[[fc:.*]]

// PerTensor-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// PerTensor-DAG: %[[q_w:.*]]= "tfl.quantize"(%[[w:.*]]) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor-DAG: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x12xf32>
// PerTensor-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensor: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w:.*]], %[[b:.*]]) {
// PerTensor-NOT: fused_activation_function = "NONE"
// PerTensor-SAME: asymmetric_quantize_inputs = true
// PerTensor: return %[[fc:.*]]
}

// CHECK-LABEL: QuantizeBatchMatmulWithActConst
// PerTensor-LABEL: QuantizeBatchMatmulWithActConst
// MinElement-LABEL: QuantizeBatchMatmulWithActConst
func.func @QuantizeBatchMatmulWithActConst(%arg0: tensor<1x3x3x512xf32>) -> tensor<1x3x3x2xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x3x3x512xf32>) -> tensor<1x3x3x512xf32>
  %w = arith.constant dense<127.0> : tensor<512x2xf32>
  %mm = "tfl.batch_matmul"(%0, %w) {adj_x = false, adj_y = false} : (tensor<1x3x3x512xf32>, tensor<512x2xf32>) -> tensor<1x3x3x2xf32>
  %mm_s = "quantfork.stats"(%mm) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
  func.return %mm_s : tensor<1x3x3x2xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x2xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x2xf32>
// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[dq_w]]) {adj_x = false, adj_y = false
// CHECK-SAME: , asymmetric_quantize_inputs = true
// CHECK: return %[[mm:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x2xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x2xf32>
// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[dq_w]]) {adj_x = false, adj_y = false
// PerTensor-SAME: , asymmetric_quantize_inputs = true
// PerTensor: return %[[mm:.*]]

// MinElement: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x2xf32>
// MinElement: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[w]]) {adj_x = false, adj_y = false} : (tensor<1x3x3x512xf32>, tensor<512x2xf32>) -> tensor<1x3x3x2xf32>
// MinElement: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeBatchMatmulWithConstAct
// PerTensor-LABEL: NotQuantizeBatchMatmulWithConstAct
func.func @NotQuantizeBatchMatmulWithConstAct(%arg0: tensor<1x1x3x512xf32>) -> tensor<1x1x12x3xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x1x3x512xf32>) -> tensor<1x1x3x512xf32>
  %w = arith.constant dense<127.0> : tensor<1x1x12x512xf32>
  %mm = "tfl.batch_matmul"(%w, %0) {adj_x = false, adj_y = true} : (tensor<1x1x12x512xf32>, tensor<1x1x3x512xf32>) -> tensor<1x1x12x3xf32>
  %mm_s = "quantfork.stats"(%mm) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x1x12x3xf32>) -> tensor<1x1x12x3xf32>
  func.return %mm_s : tensor<1x1x12x3xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x12x512xf32>
// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%[[w]], %arg0) {adj_x = false, adj_y = true}
// CHECK: return %[[mm:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x12x512xf32>
// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%[[w]], %arg0) {adj_x = false, adj_y = true}
// PerTensor: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeBatchMatmulWithActAct
// PerTensor-LABEL: NotQuantizeBatchMatmulWithActAct
func.func @NotQuantizeBatchMatmulWithActAct(%arg0: tensor<1x3x3x512xf32>) -> tensor<1x3x3x3xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x3x3x512xf32>) -> tensor<1x3x3x512xf32>
  %mm = "tfl.batch_matmul"(%0, %0) {adj_x = false, adj_y = true} : (tensor<1x3x3x512xf32>, tensor<1x3x3x512xf32>) -> tensor<1x3x3x3xf32>
  %mm_s = "quantfork.stats"(%mm) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
  func.return %mm : tensor<1x3x3x3xf32>

// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %arg0) {adj_x = false, adj_y = true}
// CHECK: return %[[mm:.*]]

// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %arg0) {adj_x = false, adj_y = true}
// PerTensor: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeConst
// Float16-LABEL: NotQuantizeConst
func.func @NotQuantizeConst() -> tensor<1x1x12x512xf32> {
  %w = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
  func.return %w : tensor<1x1x12x512xf32>

// CHECK: %[[w:.*]] = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
// CHECK: return %[[w:.*]]

// Float16: %[[w:.*]] = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
// Float16: return %[[w:.*]]
}

// CHECK-LABEL: QuantizeCustomOp
// CustomOp-LABEL: QuantizeCustomOp
// MinElement-LABEL: QuantizeCustomOp
func.func @QuantizeCustomOp(%arg0: tensor<1x1x1x1xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %w_1 = arith.constant dense<127.0> : tensor<4096x1x1x1xf32>
  %w_2 = arith.constant dense<127.0> : tensor<128x1x1x1xf32>
  %b = arith.constant dense<127.0> : tensor<2048x1x1x1xf32>
  %custom_1 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  %custom_2 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp2", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  %custom_3 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp3", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom_1, %custom_2, %custom_3 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

// CHECK: %[[w_1:.*]] = arith.constant dense<1.270000e+02> : tensor<4096x1x1x1xf32>
// CHECK: %[[w_2:.*]] = arith.constant dense<1.270000e+02> : tensor<128x1x1x1xf32>
// CHECK: %[[b:.*]] = arith.constant dense<1.270000e+02> : tensor<2048x1x1x1xf32>
// CHECK: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp2", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp3", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: return %[[custom_1:.*]], %[[custom_2:.*]], %[[custom_3:.*]]

// CustomOp-DAG: %[[w_1:.*]] = arith.constant dense<1.270000e+02> : tensor<4096x1x1x1xf32>
// CustomOp-DAG: %[[w_2:.*]] = arith.constant dense<1.270000e+02> : tensor<128x1x1x1xf32>
// CustomOp-DAG: %[[b:.*]] = arith.constant dense<1.270000e+02> : tensor<2048x1x1x1xf32>
// CustomOp-DAG: %[[q_w1:.*]] = "tfl.quantize"(%[[w_1]]) {qtype = tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>} : (tensor<4096x1x1x1xf32>) -> tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOp-DAG: %[[q_b:.*]] = "tfl.quantize"(%[[b]]) {qtype = tensor<2048x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>} : (tensor<2048x1x1x1xf32>) -> tensor<2048x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOp-DAG: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]]) : (tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<4096x1x1x1xf32>
// CustomOp: %[[dq_b:.*]] = "tfl.dequantize"(%[[q_b]]) : (tensor<2048x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<2048x1x1x1xf32>
// CustomOp: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[dq_w1]], %[[w_2]], %[[dq_b]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOp: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp2", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOp: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[dq_b]]) {custom_code = "CustomTestOp3", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOp: return %[[custom_1:.*]], %[[custom_2:.*]], %[[custom_3:.*]]

// MinElement-DAG: %[[w_1:.*]] = arith.constant dense<1.270000e+02> : tensor<4096x1x1x1xf32>
// MinElement-DAG: %[[q_w1:.*]] = "tfl.quantize"(%[[w_1]]) {qtype = tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>} : (tensor<4096x1x1x1xf32>) -> tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// MinElement-DAG: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]]) : (tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<4096x1x1x1xf32>
// MinElement-DAG: %[[w_2:.*]] = arith.constant dense<1.270000e+02> : tensor<128x1x1x1xf32>
// MinElement-DAG: %[[b:.*]] = arith.constant dense<1.270000e+02> : tensor<2048x1x1x1xf32>
// MinElement: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[dq_w1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// MinElement: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp2", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// MinElement: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp3", custom_option = #tfl<const_bytes : "0x">} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// MinElement: return %[[custom_1:.*]], %[[custom_2:.*]], %[[custom_3:.*]]
}

// CHECK-LABEL: QuantizeTransposeConvWeightOnly
// PerTensor-LABEL: QuantizeTransposeConvWeightOnly
func.func @QuantizeTransposeConvWeightOnly(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<32x4x4x128xf32>) -> tensor<32x4x4x128xf32>
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tconv = "tfl.transpose_conv"(%arg1, %w, %0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  %tconv_s = "quantfork.stats"(%tconv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  func.return %tconv_s : tensor<1x32x42x128xf32>

// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>} : (tensor<1x32x42x128xf32>) -> tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>) -> tensor<1x32x42x128xf32>
// CHECK: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w:.*]], %arg0, %[[b:.*]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: padding = "SAME"
// CHECK: return %[[tconv:.*]]

// PerTensor-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// PerTensor-DAG: %[[b:.*]]= arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>} : (tensor<1x32x42x128xf32>) -> tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1x32x42x128xf32>
// PerTensor: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w:.*]], %arg0, %[[b:.*]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: padding = "SAME"
// PerTensor: return %[[tconv:.*]]
}

// CHECK-LABEL: QuantizeGatherWeightOnly
// PerTensor-LABEL: QuantizeGatherWeightOnly
func.func @QuantizeGatherWeightOnly(%arg0: tensor<3xi32>) -> tensor<3x3x3x3xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %emb = "tfl.gather"(%w, %arg0) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<64x3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3x3xf32>
  %emb_s = "quantfork.stats"(%emb) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
  func.return %emb_s : tensor<3x3x3x3xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// CHECK: %[[emb:.*]] = "tfl.gather"(%[[dq_w]], %arg0)
// CHECK: return %[[emb:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[emb:.*]] = "tfl.gather"(%[[dq_w]], %arg0)
// PerTensor: return %[[emb:.*]]
}

// CHECK-LABEL: NotQuantizeConv3D
// PerTensor-LABEL: NotQuantizeConv3D
// Float16-LABEL: NotQuantizeConv3D
func.func @NotQuantizeConv3D(%arg0: tensor<?x28x28x28x8xf32>) -> tensor<?x26x26x26x16xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<?x28x28x28x8xf32>) -> tensor<?x28x28x28x8xf32>
  %cst = arith.constant dense<16> : tensor<1xi64>
  %cst_0 = "tfl.no_value"() {value = unit} : () -> none
  %w = arith.constant dense<127.0> : tensor<3x3x3x8x16xf32>
  %b = arith.constant dense<0.0> : tensor<16xf32>
  %conv = "tfl.conv_3d"(%0, %w, %cst_0) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
  %conv_s = "quantfork.stats"(%conv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  %1 = "tfl.shape"(%conv_s) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
  %2 = "tfl.broadcast_args"(%1, %cst) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
  %broad1 = "tfl.broadcast_to"(%conv_s, %2) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
  %broad2 = "tfl.broadcast_to"(%b, %2) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
  %broad1_s = "quantfork.stats"(%broad1) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  %broad2_s = "quantfork.stats"(%broad2) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  %add = "tfl.add"(%broad1_s, %broad2_s) {fused_activation_function = "RELU"} : (tensor<?x26x26x26x16xf32>, tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  %add_s = "quantfork.stats"(%add) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  func.return %add_s : tensor<?x26x26x26x16xf32>

// CHECK-DAG: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// CHECK-DAG: %[[const:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf32>
// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[const]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// CHECK: %2 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// CHECK: %3 = "tfl.broadcast_args"(%2, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// CHECK: %4 = "tfl.broadcast_to"(%[[conv3d]], %3) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// CHECK: %5 = "tfl.broadcast_to"(%[[b:.*]], %3) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// CHECK: %6 = tfl.add %4, %5 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// CHECK: return %6 : tensor<?x26x26x26x16xf32>

// PerTensor: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// PerTensor: %[[const:.*]] = "tfl.no_value"() {value} : () -> none
// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// PerTensor: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %[[const]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// PerTensor: %2 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// PerTensor: %3 = "tfl.broadcast_args"(%2, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// PerTensor: %4 = "tfl.broadcast_to"(%[[conv3d]], %3) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// PerTensor: %5 = "tfl.broadcast_to"(%[[b:.*]], %3) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// PerTensor: %6 = tfl.add %4, %5 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// PerTensor: return %6 : tensor<?x26x26x26x16xf32>

// Float16-DAG: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// Float16-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf16>
// Float16-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf16>
// Float16-DAG: %[[const:.*]] = "tfl.no_value"() {value} : () -> none
// Float16-DAG: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<3x3x3x8x16xf16>) -> tensor<3x3x3x8x16xf32>
// Float16-DAG: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<16xf16>) -> tensor<16xf32>
// Float16: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[dq_w]], %[[const]]) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// Float16: %4 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// Float16: %5 = "tfl.broadcast_args"(%4, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// Float16: %6 = "tfl.broadcast_to"(%[[conv3d]], %5) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// Float16: %7 = "tfl.broadcast_to"(%[[dq_b:.*]], %5) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// Float16: %8 = tfl.add %6, %7 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// Float16: return %8 : tensor<?x26x26x26x16xf32>
}

// CHECK-LABEL: QuantizeMultiUses
// PerTensor-LABEL: QuantizeMultiUses
// Float16-LABEL: QuantizeMultiUses
func.func @QuantizeMultiUses(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x112xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %conv_s = "quantfork.stats"(%conv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %dconv_s = "quantfork.stats"(%dconv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %bmm = "tfl.batch_matmul"(%conv_s, %dconv_s) {adj_x = false, adj_y = true} : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x112xf32>
  %bmm_s = "quantfork.stats"(%bmm) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x112xf32>) -> tensor<1x112x112x112xf32>
  func.return %bmm_s : tensor<1x112x112x112xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK-DAG: %[[q_w1:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK-DAG: %[[q_w2:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// CHECK-DAG: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]])
// CHECK-DAG: %[[dq_w2:.*]] = "tfl.dequantize"(%[[q_w2]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w2]], %[[b]])
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w1]], %[[b]])
// CHECK: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// CHECK-NOT: , asymmetric_quantize_inputs = true
// CHECK-SAME: }
// CHECK: return %[[bmm:.*]]

// PerTensor-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor-DAG: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]])
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[b]])
// PerTensor: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// PerTensor-NOT: , asymmetric_quantize_inputs = true
// PerTensor-SAME: }
// PerTensor: return %[[bmm:.*]]

// Float16-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16-DAG: %[[b:.*]] = arith.constant dense<-1.237300e+00> : tensor<64xf16>
// Float16-DAG: %[[dq_w:.*]] = "tfl.dequantize"(%[[w:.*]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16-DAG: %[[dq_b:.*]] = "tfl.dequantize"(%[[b:.*]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[dq_b]])
// Float16: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[dq_b]])
// Float16: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// Float16: return %[[bmm:.*]]
}

// Float16-LABEL: LargeFloat16Constants
func.func @LargeFloat16Constants(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<7.270000e+04> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-8.0e+4> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %conv_s = "quantfork.stats"(%conv) {layerStats = dense<[0.000000e+00, 1.000000e+01]> : tensor<2xf32>} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  func.return %conv_s : tensor<1x112x112x64xf32>

// Float16-DAG: %[[w:.*]] = arith.constant dense<6.550400e+04> : tensor<64x3x3x3xf16>
// Float16-DAG: %[[b:.*]] = arith.constant dense<-6.550400e+04> : tensor<64xf16>
}

// LSTMOpQuantized-LABEL: LSTMOpNotPartiallyQuantized
// LSTMOpNotQuantized-LABEL: LSTMOpNotPartiallyQuantized
func.func @LSTMOpNotPartiallyQuantized(%arg0: tensor<1x28x28xf32>) -> tensor<1x28x20xf32> {
    %cst_2 = "tfl.no_value"() {value = unit} : () -> none
    %cst_3 = arith.constant dense<1.0> : tensor<20x20xf32>
    %cst_7 = arith.constant dense<1.0> : tensor<20xf32>
    %recurrent_input = arith.constant dense<1.0> : tensor<1x20xf32>
    %recurrent_stats = "quantfork.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x20xf32>) -> tensor<1x20xf32>
    %cell_input = arith.constant dense<1.0> : tensor<1x20xf32>
    %cell_stats = "quantfork.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x20xf32>) -> tensor<1x20xf32>
    %0 = "tfl.unidirectional_sequence_lstm"(%arg0,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_7, %cst_7, %cst_7,
      %cst_7, %cst_7, %cst_7, %cst_7,
      %cst_3, %cst_2,
      %recurrent_stats, %cell_stats,
      %cst_2, %cst_2, %cst_2, %cst_2) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", proj_clip = 0.000000e+00 : f32, time_major = false}
    : ( tensor<1x28x28xf32>,
        tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>,
        tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>,
        tensor<20xf32>, tensor<20xf32>, tensor<20xf32>,
        tensor<20xf32>, tensor<20xf32>, tensor<20xf32>, tensor<20xf32>,
        tensor<20x20xf32>, none,
        tensor<1x20xf32>, tensor<1x20xf32>,
        none, none, none, none) -> tensor<1x28x20xf32>
    %1 = "quantfork.stats"(%0) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<1x28x20xf32>) -> tensor<1x28x20xf32>
    func.return %1 : tensor<1x28x20xf32>

// LSTMOpQuantized-DAG: %[[dq1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<20x20x!quant.uniform<i8<-127:127>:f32, 0.0078740157480314959>>) -> tensor<20x20xf32>
// LSTMOpQuantized-DAG: %[[dq3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<20x!quant.uniform<i8<-127:127>:f32, 0.0078740157480314959>>) -> tensor<20xf32>
// LSTMOpQuantized: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%arg0, %[[dq1]], %[[dq1]], %[[dq1]], %[[dq1]], %[[dq1]], %[[dq1]], %[[dq1]], %[[dq1]], %[[dq3]], %[[dq3]], %[[dq3]], %cst_0, %cst_0, %cst_0, %cst_0, %[[dq1]], %0, %cst_1, %cst_1, %0, %0, %0, %0)

// LSTMOpNotQuantized-DAG: %[[cst_1:.*]] = arith.constant dense<1.000000e+00> : tensor<20x20xf32>
// LSTMOpNotQuantized-DAG: %[[cst_3:.*]] = arith.constant dense<1.000000e+00> : tensor<20xf32>
// LSTMOpNotQuantized: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%arg0, %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_1]], %[[cst_3]], %[[cst_3]], %[[cst_3]], %cst_0, %cst_0, %cst_0, %cst_0, %[[cst_1]], %0, %cst_1, %cst_1, %0, %0, %0, %0)
}
