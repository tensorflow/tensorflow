// RUN: stablehlo-quant-opt %s -split-input-file -optimize-graph | FileCheck %s

// CHECK-LABEL: @merge_requantization_followed_by_dequantization
// CHECK-SAME: %[[ARG_0:.*]]: tensor<1x3x4x3xf32>
func.func @merge_requantization_followed_by_dequantization(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<4.000000e-01> : tensor<2x3x3x2xf32>
  // CHECK: %[[QUANT_CST:.*]] = stablehlo.uniform_quantize %[[CST]]
  // CHECK: %[[QUANT_ARG_0:.*]] = stablehlo.uniform_quantize %[[ARG_0]]
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[QUANT_ARG_0]], %[[QUANT_CST]])
  // CHECK-NOT: stablehlo.uniform_quantize
  // CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]]
  // CHECK: return %[[DEQUANT]]
  %cst = stablehlo.constant dense<0.4> : tensor<2x3x3x2xf32>
  %quant_cst = stablehlo.uniform_quantize %cst : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.015>>
  %quant_arg = stablehlo.uniform_quantize %arg0 : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039207626791561354:-128>>
  %conv = stablehlo.convolution(%quant_arg, %quant_cst) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039207626791561354:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.015>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, 5.8949912267181218E-5>>
  %requant = stablehlo.uniform_quantize %conv : (tensor<1x3x4x2x!quant.uniform<i32:f32, 5.8949912267181218E-5>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, 0.045673100153605144:-62>>
  %dequant = stablehlo.uniform_dequantize %requant : (tensor<1x3x4x2x!quant.uniform<i8:f32, 0.045673100153605144:-62>>) -> tensor<1x3x4x2xf32>
  func.return %dequant : tensor<1x3x4x2xf32>
}

// -----

// CHECK-LABEL: @dont_merge_quantization_followed_by_quantization
// CHECK-SAME: %[[ARG_0:.*]]: tensor<1x3x4x3xf32>
func.func @dont_merge_quantization_followed_by_quantization(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32> {
  // CHECK: %[[QUANT_ARG_0:.*]] = stablehlo.uniform_quantize %[[ARG_0]]
  // CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[QUANT_ARG_0]]
  // CHECK: return %[[DEQUANT]]
  %quant_arg = stablehlo.uniform_quantize %arg0 : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039207626791561354:-128>>
  %dequant = stablehlo.uniform_dequantize %quant_arg : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039207626791561354:-128>>) -> tensor<1x3x4x3xf32>
  func.return %dequant : tensor<1x3x4x3xf32>
}
