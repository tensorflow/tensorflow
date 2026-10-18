// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: litert-opt %s -tfl-optimize -tfl-propagate-qparams -tfl-bias-quantizer -tfl-fuse-qdq -canonicalize | FileCheck %s

// CHECK-LABEL: fuseMulIntoPerTensorConv2dWithQDQs
func.func @fuseMulIntoPerTensorConv2dWithQDQs(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x8x7x3xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %w = arith.constant dense<2.0> : tensor<3x3x3x3xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>
  %dq = "tfl.dequantize"(%q) : (tensor<3x3x3x3x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<3x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %dq, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x3xf32>, tensor<f32>) -> tensor<256x8x7x3xf32>
  func.return %1 : tensor<256x8x7x3xf32>

  // CHECK: %[[bias:.*]] = arith.constant dense<[1.500000e+00, 3.000000e+00, 4.500000e+00]> : tensor<3xf32>
  // CHECK: %[[qweight:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x3x3x3x!quant.uniform<i8:f32, 0.15000000000000002:1>>, value = dense<21> : tensor<3x3x3x3xi8>}>
  // CHECK: %[[weight:.*]] = "tfl.dequantize"(%[[qweight]])
  // CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[weight]], %[[bias]])
  // CHECK: return %[[conv]] : tensor<256x8x7x3xf32>
}

// CHECK-LABEL: fuseMulIntoPerAxisConv2dWithQDQsBroadcastScalar
func.func @fuseMulIntoPerAxisConv2dWithQDQsBroadcastScalar(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x8x7x3xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %w = arith.constant dense<2.0> : tensor<3x3x3x3xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<3x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3}>>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3}>>
  %dq = "tfl.dequantize"(%q) : (tensor<3x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3}>>) -> tensor<3x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %dq, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x3xf32>, tensor<f32>) -> tensor<256x8x7x3xf32>
  func.return %1 : tensor<256x8x7x3xf32>

  // CHECK: %[[bias:.*]] = arith.constant dense<[1.500000e+00, 3.000000e+00, 4.500000e+00]> : tensor<3xf32>
  // CHECK: %[[qweight:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x3x3x3x!quant.uniform<i8:f32:0, {0.15000000000000002,0.30000000000000004,0.44999999999999996}>>, value = dense<20> : tensor<3x3x3x3xi8>}>
  // CHECK: %[[weight:.*]] = "tfl.dequantize"(%[[qweight]])
  // CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[weight]], %[[bias]])
  // CHECK: return %[[conv]] : tensor<256x8x7x3xf32>
}
