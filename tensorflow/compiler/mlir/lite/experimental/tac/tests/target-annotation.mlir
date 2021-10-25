// RUN: tac-opt-all-backends -tfl-target-annotation='device-specs=GPU' %s -split-input-file -verify-diagnostics | FileCheck %s

func @testConv(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// -----

func @testDepthwiseConv(%arg0: tensor<1x112x112x32xf32>, %arg1: tensor<1x3x3x32xf32>, %arg2: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x32xf32>, tensor<1x3x3x32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

// -----

func @testAvgPool(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// -----

func @testMaxPool(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// -----

func @testAddReluPack(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
   // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
   // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %1 = "tfl.add"(%arg0, %0) {fused_activation_function = "RELU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
   // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %2 = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK: tac.device = "CPU", tac.inference_type = "FLOAT"
  %3 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return
}

func @notAnnotateConst(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  // CHECK-NOT: tac.device tac.inference_type
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
  // CHECK-NOT: tac.device tac.inference_type
  %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  // CHECK: tac.device = "GPU", tac.inference_type = "FLOAT"
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %2 : tensor<256x30x30x16xf32>
}

// -----

func @notAnnotateQuantizeDequantize(%arg0: tensor<4x384x32x!quant.uniform<i8:f32, 0.2:-3>>) -> tensor<4x384x32xf32> {
  // CHECK-NOT: tac.device tac.inference_type
  %0 = "tfl.pseudo_const"() {value = dense<[1, 4, 384, 32]> : tensor<4xi32>} : () -> tensor<4xi32>
  // CHECK-NOT: tac.device tac.inference_type
  %1 = "tfl.pseudo_const"() {value = dense<[4, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"
  %2 = "tfl.reshape"(%arg0, %0) : (tensor<4x384x32x!quant.uniform<i8:f32, 0.2:-3>>, tensor<4xi32>) -> tensor<1x4x384x32x!quant.uniform<i8:f32, 0.2:-3>>
  // CHECK-NOT: tac.device tac.inference_type
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x4x384x32x!quant.uniform<i8:f32, 0.19:1>>} : (tensor<1x4x384x32x!quant.uniform<i8:f32, 0.2:-3>>) -> tensor<1x4x384x32x!quant.uniform<i8:f32, 0.19:1>>
  // CHECK: tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"
  %4 = "tfl.reshape"(%3, %1) : (tensor<1x4x384x32x!quant.uniform<i8:f32, 0.19:1>>, tensor<3xi32>) -> tensor<4x384x32x!quant.uniform<i8:f32, 0.19:1>>
  // CHECK-NOT: tac.device tac.inference_type
  %5 = "tfl.dequantize"(%4) : (tensor<4x384x32x!quant.uniform<i8:f32, 0.19:1>>) -> tensor<4x384x32xf32>
  return %5 : tensor<4x384x32xf32>

}

func @annotateInferenceType(%arg0: tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>{
  // CHECK-NOT: tac.device tac.inference_type
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1x384x1xi8>} : () -> tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>
  // CHECK: tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  return %1 : tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
}
