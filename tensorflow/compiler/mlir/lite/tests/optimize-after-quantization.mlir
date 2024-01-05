// RUN: tf-opt %s -tfl-prepare-quantize -canonicalize -tfl-quantize -canonicalize -tfl-optimize -canonicalize | FileCheck %s

// CHECK-LABEL: fuseMulIntoPerTensorConv2dWithQDQs
func.func @fuseMulIntoPerTensorConv2dWithQDQs(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x8x7x3xf32> {
  %cst = arith.constant dense<1.5> : tensor<3xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %w = arith.constant dense<2.0> : tensor<3x3x3x3xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>
  %dq = "tfl.dequantize"(%q) : (tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<3x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %dq, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  func.return %1 : tensor<256x8x7x3xf32>

  // CHECK: %[[weight:.*]] = arith.constant dense<3.000000e+00> : tensor<3x3x3x3xf32>
  // CHECK: %[[bias:.*]] = arith.constant dense<[1.500000e+00, 3.000000e+00, 4.500000e+00]>
  // CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[weight]], %[[bias]])
  // CHECK: return %[[conv]] : tensor<256x8x7x3xf32>
}
