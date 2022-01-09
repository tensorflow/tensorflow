// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range  --tfl-enable-dynamic-range-per-channel-quantization=false | FileCheck --check-prefix=PerTensor %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range  --tfl-min-elements-for-weights=10000 | FileCheck --check-prefix=MinElement %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range  --tfl-enable-float16-quantization | FileCheck --check-prefix=Float16 %s

// CHECK-LABEL: QuantizeConv2D
// PerTensor-LABEL: QuantizeConv2D
// MinElement-LABEL: QuantizeConv2D
// Float16-LABEL: QuantizeConv2D
func @QuantizeConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %conv : tensor<1x112x112x64xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]])
// CHECK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: dilation_h_factor = 1 : i32
// CHECK: return %[[conv:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: dilation_h_factor = 1 : i32
// PerTensor: return %[[conv:.*]]

// MinElement: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// MinElement: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// MinElement: %[[conv:.*]]= "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// MinElement: return %[[conv:.*]]

// Float16: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16: %[[b:.*]] = arith.constant dense<-1.237300e+00> : tensor<64xf16>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[conv:.*]]= "tfl.conv_2d"(%arg0, %[[dq_w]], %[[dq_b]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// Float16: return %[[conv:.*]]
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// PerTensor-LABEL: QuantizeDepthwiseConv2D
// MinElement-LABEL: QuantizeDepthwiseConv2D
// Float16-LABEL: QuantizeDepthwiseConv2D
func @QuantizeDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<127.0> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %dconv : tensor<1x112x112x64xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]])
// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: depth_multiplier = 4 : i32
// CHECK: return %[[dconv:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[b]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: depth_multiplier = 4 : i32
// PerTensor: return %[[dconv:.*]]

// MinElement: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// MinElement: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
// MinElement: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// MinElement: return %[[dconv:.*]]

// Float16: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf16>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[dq_b]]) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
// Float16: return %[[dconv:.*]]
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x512xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %b = arith.constant dense<0.0> : tensor<512xf32>
  %fc = "tfl.fully_connected"(%arg0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<512x12xf32>, tensor<512xf32>) -> tensor<1x112x112x512xf32>
  return %fc : tensor<1x112x112x512xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x12xf32>
// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w]], %[[b]]) {
// CHECK-NOT: fused_activation_function = "NONE"
// CHECK-SAME: asymmetric_quantize_inputs = true
// CHECK: return %[[fc:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// PerTensor: %[[q_w:.*]]= "tfl.quantize"(%[[w:.*]]) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x12xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensor: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq_w:.*]], %[[b:.*]]) {
// PerTensor-NOT: fused_activation_function = "NONE"
// PerTensor-SAME: asymmetric_quantize_inputs = true
// PerTensor: return %[[fc:.*]]
}

// CHECK-LABEL: QuantizeBatchMatmulWithActConst
// PerTensor-LABEL: QuantizeBatchMatmulWithActConst
// MinElement-LABEL: QuantizeBatchMatmulWithActConst
func @QuantizeBatchMatmulWithActConst(%arg0: tensor<1x3x3x512xf32>) -> tensor<1x3x3x12xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %mm = "tfl.batch_matmul"(%arg0, %w) {adj_x = false, adj_y = false} : (tensor<1x3x3x512xf32>, tensor<512x12xf32>) -> tensor<1x3x3x12xf32>
  return %mm : tensor<1x3x3x12xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x12x!quant.uniform<i8:f32, 0.49803921568627452:-128>>}
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x12x!quant.uniform<i8:f32, 0.49803921568627452:-128>>) -> tensor<512x12xf32>
// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[dq_w]]) {adj_x = false, adj_y = false
// CHECK-SAME: , asymmetric_quantize_inputs = true
// CHECK: return %[[mm:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<512x12x!quant.uniform<i8:f32, 0.49803921568627452:-128>>}
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<512x12x!quant.uniform<i8:f32, 0.49803921568627452:-128>>) -> tensor<512x12xf32>
// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[dq_w]]) {adj_x = false, adj_y = false
// PerTensor-SAME: , asymmetric_quantize_inputs = true
// PerTensor: return %[[mm:.*]]

// MinElement: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// MinElement: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %[[w]]) {adj_x = false, adj_y = false} : (tensor<1x3x3x512xf32>, tensor<512x12xf32>) -> tensor<1x3x3x12xf32>
// MinElement: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeBatchMatmulWithConstAct
// PerTensor-LABEL: NotQuantizeBatchMatmulWithConstAct
func @NotQuantizeBatchMatmulWithConstAct(%arg0: tensor<1x1x3x512xf32>) -> tensor<1x1x12x3xf32> {
  %w = arith.constant dense<127.0> : tensor<1x1x12x512xf32>
  %mm = "tfl.batch_matmul"(%w, %arg0) {adj_x = false, adj_y = true} : (tensor<1x1x12x512xf32>, tensor<1x1x3x512xf32>) -> tensor<1x1x12x3xf32>
  return %mm : tensor<1x1x12x3xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x12x512xf32>
// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%[[w]], %arg0) {adj_x = false, adj_y = true}
// CHECK: return %[[mm:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x1x12x512xf32>
// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%[[w]], %arg0) {adj_x = false, adj_y = true}
// PerTensor: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeBatchMatmulWithActAct
// PerTensor-LABEL: NotQuantizeBatchMatmulWithActAct
func @NotQuantizeBatchMatmulWithActAct(%arg0: tensor<1x3x3x512xf32>) -> tensor<1x3x3x3xf32> {
  %mm = "tfl.batch_matmul"(%arg0, %arg0) {adj_x = false, adj_y = true} : (tensor<1x3x3x512xf32>, tensor<1x3x3x512xf32>) -> tensor<1x3x3x3xf32>
  return %mm : tensor<1x3x3x3xf32>

// CHECK: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %arg0) {adj_x = false, adj_y = true}
// CHECK: return %[[mm:.*]]

// PerTensor: %[[mm:.*]] = "tfl.batch_matmul"(%arg0, %arg0) {adj_x = false, adj_y = true}
// PerTensor: return %[[mm:.*]]
}

// CHECK-LABEL: NotQuantizeConst
// Float16-LABEL: NotQuantizeConst
func @NotQuantizeConst() -> tensor<1x1x12x512xf32> {
  %w = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
  return %w : tensor<1x1x12x512xf32>

// CHECK: %[[w:.*]] = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
// CHECK: return %[[w:.*]]

// Float16: %[[w:.*]] = arith.constant dense<-1.23697901> : tensor<1x1x12x512xf32>
// Float16: return %[[w:.*]]
}

// CHECK-LABEL: QuantizeTransposeConvWeightOnly
// PerTensor-LABEL: QuantizeTransposeConvWeightOnly
func @QuantizeTransposeConvWeightOnly(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tconv = "tfl.transpose_conv"(%arg1, %w, %arg0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  return %tconv : tensor<1x32x42x128xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// CHECK: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>} : (tensor<1x32x42x128xf32>) -> tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00}>>) -> tensor<1x32x42x128xf32>
// CHECK: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// CHECK: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w:.*]], %arg0, %[[b:.*]]) {
// CHECK-NOT: asymmetric_quantize_inputs = true
// CHECK-SAME: padding = "SAME"
// CHECK: return %[[tconv:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// PerTensor: %[[q_w:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>} : (tensor<1x32x42x128xf32>) -> tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// PerTensor: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w]]) : (tensor<1x32x42x128x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1x32x42x128xf32>
// PerTensor: %[[b:.*]]= arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensor: %[[tconv:.*]] = "tfl.transpose_conv"(%arg1, %[[dq_w:.*]], %arg0, %[[b:.*]]) {
// PerTensor-NOT: asymmetric_quantize_inputs = true
// PerTensor-SAME: padding = "SAME"
// PerTensor: return %[[tconv:.*]]
}

// CHECK-LABEL: NotQuantizeConv3D
// PerTensor-LABEL: NotQuantizeConv3D
// Float16-LABEL: NotQuantizeConv3D
func @NotQuantizeConv3D(%arg0: tensor<?x28x28x28x8xf32>) -> tensor<?x26x26x26x16xf32> {
  %cst = arith.constant dense<16> : tensor<1xi64>
  %cst_0 = constant unit
  %w = arith.constant dense<127.0> : tensor<3x3x3x8x16xf32>
  %b = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "tfl.conv_3d"(%arg0, %w, %cst_0) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
  %1 = "tfl.shape"(%0) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
  %2 = "tfl.broadcast_args"(%1, %cst) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
  %3 = "tfl.broadcast_to"(%0, %2) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
  %4 = "tfl.broadcast_to"(%b, %2) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
  %5 = "tfl.add"(%3, %4) {fused_activation_function = "RELU"} : (tensor<?x26x26x26x16xf32>, tensor<?x26x26x26x16xf32>) -> tensor<?x26x26x26x16xf32>
  return %5 : tensor<?x26x26x26x16xf32>

// CHECK: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// CHECK: %[[const:.*]] = constant unit
// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf32>
// CHECK-DAG: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %cst_0) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// CHECK: %1 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// CHECK: %2 = "tfl.broadcast_args"(%1, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// CHECK: %3 = "tfl.broadcast_to"(%[[conv3d]], %2) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// CHECK: %4 = "tfl.broadcast_to"(%[[b:.*]], %2) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// CHECK: %5 = tfl.add %3, %4 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// CHECK: return %5 : tensor<?x26x26x26x16xf32>

// PerTensor: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// PerTensor: %[[const:.*]] = constant unit
// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// PerTensor: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[w]], %cst_0) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// PerTensor: %1 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// PerTensor: %2 = "tfl.broadcast_args"(%1, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// PerTensor: %3 = "tfl.broadcast_to"(%[[conv3d]], %2) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// PerTensor: %4 = "tfl.broadcast_to"(%[[b:.*]], %2) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// PerTensor: %5 = tfl.add %3, %4 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// PerTensor: return %5 : tensor<?x26x26x26x16xf32>

// Float16: %[[out_ch:.*]] = arith.constant dense<16> : tensor<1xi64>
// Float16: %[[const:.*]] = constant unit
// Float16: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<3x3x3x8x16xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w]]) : (tensor<3x3x3x8x16xf16>) -> tensor<3x3x3x8x16xf32>
// Float16: %[[b:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf16>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b]]) : (tensor<16xf16>) -> tensor<16xf32>
// Float16: %[[conv3d:.*]] = "tfl.conv_3d"(%arg0, %[[dq_w]], %cst_0) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x28x28x28x8xf32>, tensor<3x3x3x8x16xf32>, none) -> tensor<?x26x26x26x16xf32>
// Float16: %3 = "tfl.shape"(%[[conv3d]]) : (tensor<?x26x26x26x16xf32>) -> tensor<5xi64>
// Float16: %4 = "tfl.broadcast_args"(%3, %[[out_ch]]) : (tensor<5xi64>, tensor<1xi64>) -> tensor<5xi64>
// Float16: %5 = "tfl.broadcast_to"(%[[conv3d]], %4) : (tensor<?x26x26x26x16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// Float16: %6 = "tfl.broadcast_to"(%[[dq_b:.*]], %4) : (tensor<16xf32>, tensor<5xi64>) -> tensor<?x26x26x26x16xf32>
// Float16: %7 = tfl.add %5, %6 {fused_activation_function = "RELU"} : tensor<?x26x26x26x16xf32>
// Float16: return %7 : tensor<?x26x26x26x16xf32>
}

// CHECK-LABEL: QuantizeMultiUses
// PerTensor-LABEL: QuantizeMultiUses
// Float16-LABEL: QuantizeMultiUses
func @QuantizeMultiUses(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x122xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %conv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dconv = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %bmm = "tfl.batch_matmul"(%conv, %dconv) {adj_x = false, adj_y = true} : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x122xf32>
  return %bmm : tensor<1x112x112x122xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %[[q_w1:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]])
// CHECK: %[[q_w2:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// CHECK: %[[dq_w2:.*]] = "tfl.dequantize"(%[[q_w2]])
// CHECK: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w2]], %[[b]])
// CHECK: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w1]], %[[b]])
// CHECK: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// CHECK-NOT: , asymmetric_quantize_inputs = true
// CHECK-SAME: }
// CHECK: return %[[bmm:.*]]

// PerTensor: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %[[q_w1:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[q_w2:.*]] = "tfl.quantize"(%[[w]]) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %[[dq_w2:.*]] = "tfl.dequantize"(%[[q_w2]]) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %[[b:.*]] = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w2]], %[[b]])
// PerTensor: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w1]], %[[b]])
// PerTensor: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// PerTensor-NOT: , asymmetric_quantize_inputs = true
// PerTensor-SAME: }
// PerTensor: return %[[bmm:.*]]

// Float16: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf16>
// Float16: %[[dq_w:.*]] = "tfl.dequantize"(%[[w:.*]]) : (tensor<64x3x3x3xf16>) -> tensor<64x3x3x3xf32>
// Float16: %[[b:.*]] = arith.constant dense<-1.237300e+00> : tensor<64xf16>
// Float16: %[[dq_b:.*]] = "tfl.dequantize"(%[[b:.*]]) : (tensor<64xf16>) -> tensor<64xf32>
// Float16: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq_w]], %[[dq_b]])
// Float16: %[[dconv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[dq_w]], %[[dq_b]])
// Float16: %[[bmm:.*]] = "tfl.batch_matmul"(%[[conv]], %[[dconv]]) {adj_x = false, adj_y = true
// Float16: return %[[bmm:.*]]
}
