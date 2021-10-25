// RUN: tf-opt %s -tfl-prepare-dynamic-quantize | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-dynamic-quantize -tfl-dynamic-quantize-disable-per-channel | FileCheck --check-prefix=PerTensor %s

// CHECK-LABEL: QuantizeConv2D
// PerTensor-LABEL: QuantizeConv2D
func @QuantizeConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %cv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %cv : tensor<1x112x112x64xf32>

// CHECK: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK: %1 = "tfl.dequantize"(%0)
// CHECK: %cst_0 = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %2 = "tfl.conv_2d"(%arg0, %1, %cst_0)
// CHECK: return %2

// PerTensor: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %1 = "tfl.dequantize"(%0) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %cst_0 = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %2 = "tfl.conv_2d"(%arg0, %1, %cst_0)
// PerTensor: return %2
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// PerTensor-LABEL: QuantizeDepthwiseConv2D
func @QuantizeDepthwiseConv2D(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
  %w = arith.constant dense<127.0> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<64xf32>
  %dc = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  return %dc : tensor<1x112x112x64xf32>

// CHECK: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %1 = "tfl.dequantize"(%0)
// CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
// CHECK: %2 = "tfl.depthwise_conv_2d"(%arg0, %1, %cst_0)
// CHECK: return %2

// PerTensor: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %1 = "tfl.dequantize"(%0) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
// PerTensor: %2 = "tfl.depthwise_conv_2d"(%arg0, %1, %cst_0)
// PerTensor: return %2
}

// CHECK-LABEL: QuantizeFullyConnected
// PerTensor-LABEL: QuantizeFullyConnected
func @QuantizeFullyConnected(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x512xf32> {
  %w = arith.constant dense<127.0> : tensor<512x12xf32>
  %b = arith.constant dense<0.0> : tensor<512xf32>
  %fc = "tfl.fully_connected"(%arg0, %w, %b) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<512x12xf32>, tensor<512xf32>) -> tensor<1x112x112x512xf32>
  return %fc : tensor<1x112x112x512xf32>

// CHECK: %cst = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x12xf32>
// CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
// CHECK: %2 = "tfl.fully_connected"(%arg0, %1, %cst_0

// PerTensor: %cst = arith.constant dense<1.270000e+02> : tensor<512x12xf32>
// PerTensor: %0 = "tfl.quantize"(%cst) {qtype = tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %1 = "tfl.dequantize"(%0) : (tensor<512x12x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<512x12xf32>
// PerTensor: %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
// PerTensor: %2 = "tfl.fully_connected"(%arg0, %1, %cst_0
}

// CHECK-LABEL: QuantizeTransposeConv
// PerTensor-LABEL: QuantizeTransposeConv
func @QuantizeTransposeConv(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<4xi32>) -> tensor<1x32x42x128xf32> {
  %w = arith.constant dense<127.0> : tensor<1x32x42x128xf32>
  %b = arith.constant dense<0.0> : tensor<1x32x42x128xf32>
  %tc = "tfl.transpose_conv"(%arg1, %w, %arg0, %b) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<1x32x42x128xf32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>) -> tensor<1x32x42x128xf32>
  return %tc : tensor<1x32x42x128xf32>

// CHECK: %cst = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// CHECK: %0 = "tfl.transpose_conv"(%arg1, %cst, %arg0, %cst_0)
// CHECK: return %0

// PerTensor: %cst = arith.constant dense<1.270000e+02> : tensor<1x32x42x128xf32>
// PerTensor: %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x32x42x128xf32>
// PerTensor: %0 = "tfl.transpose_conv"(%arg1, %cst, %arg0, %cst_0)
// PerTensor: return %0
}

// CHECK-LABEL: QuantizeMultiUses
// PerTensor-LABEL: QuantizeMultiUses
func @QuantizeMultiUses(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x122xf32> {
  %w = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<-1.23697901> : tensor<64xf32>
  %cv = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %dc = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %bm = "tfl.batch_matmul"(%cv, %dc) {adj_x = false, adj_y = true} : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x122xf32>
  return %bm : tensor<1x112x112x122xf32>

// CHECK: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:3, {1.000000e+00,1.000000e+00,1.000000e+00}
// CHECK: %1 = "tfl.dequantize"(%0)
// CHECK: %2 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e+00,1.000000e+00,1.000000e+00
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: %cst_0 = arith.constant dense<-1.23697901> : tensor<64xf32>
// CHECK: %4 = "tfl.conv_2d"(%arg0, %3, %cst_0)
// CHECK: %5 = "tfl.depthwise_conv_2d"(%arg0, %1, %cst_0)
// CHECK: %6 = "tfl.batch_matmul"(%4, %5)
// CHECK: return %6

// PerTensor: %cst = arith.constant dense<1.270000e+02> : tensor<64x3x3x3xf32>
// PerTensor: %0 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %1 = "tfl.dequantize"(%0) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %2 = "tfl.quantize"(%cst) {qtype = tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>}
// PerTensor: %3 = "tfl.dequantize"(%2) : (tensor<64x3x3x3x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<64x3x3x3xf32>
// PerTensor: %cst_0 = arith.constant dense<-1.23697901> : tensor<64xf32>
// PerTensor: %4 = "tfl.conv_2d"(%arg0, %3, %cst_0)
// PerTensor: %5 = "tfl.depthwise_conv_2d"(%arg0, %1, %cst_0)
// PerTensor: %6 = "tfl.batch_matmul"(%4, %5)
// PerTensor: return %6
}
