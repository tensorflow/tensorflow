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
// RUN: litert-opt %s -tfl-bias-quantizer -tfl-fuse-qdq | FileCheck %s

// Checks that BiasQuantizerPass and FuseQDQPass agree on which
// tfl.fully_connected ops get a blockwise quantized filter and an int32 bias.

// CHECK-LABEL: Int8SrqFullyConnectedBlockwiseFilterWithBias
func.func @Int8SrqFullyConnectedBlockwiseFilterWithBias(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-DAG: %[[W:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = {{.*}} : tensor<4x1xf32>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x!quant.uniform<i32:f32:0, {2.500000e-01,5.000000e-01,7.500000e-01,1.250000e-01}>>
  // CHECK: "tfl.fully_connected"(%{{.*}}, %[[W]], %[[B]]) {{.*}} -> tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01:1>>
  // CHECK-NOT: "tfl.blockwise_dequantize"
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense_resource<blockwise_scales> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 8], symmetric = false} : (tensor<4x8xi2>, tensor<4x1xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_q = "tfl.quantize"(%arg0) <{qtype = tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>}> : (tensor<2x8xf32>) -> tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>
  %in_dq = "tfl.dequantize"(%in_q) : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  %out_dq = "tfl.dequantize"(%out_q) : (tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>) -> tensor<2x4xf32>
  func.return %out_dq : tensor<2x4xf32>
}

// CHECK-LABEL: Int16SrqFullyConnectedBlockwiseFilterKeepsFloatBias
func.func @Int16SrqFullyConnectedBlockwiseFilterKeepsFloatBias(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: !tfl.blockwise_quant
  // CHECK-NOT: i32:f32
  // CHECK: %[[W_DQ:.*]] = "tfl.blockwise_dequantize"
  // CHECK: "tfl.fully_connected"(%{{.*}}, %[[W_DQ]], %{{.*}}) {{.*}} : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<0.5> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 8], symmetric = false} : (tensor<4x8xi2>, tensor<4x1xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_q = "tfl.quantize"(%arg0) <{qtype = tensor<2x8x!quant.uniform<i16:f32, 0.5>>}> : (tensor<2x8xf32>) -> tensor<2x8x!quant.uniform<i16:f32, 0.5>>
  %in_dq = "tfl.dequantize"(%in_q) : (tensor<2x8x!quant.uniform<i16:f32, 0.5>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i16:f32, 0.25>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i16:f32, 0.25>>
  %out_dq = "tfl.dequantize"(%out_q) : (tensor<2x4x!quant.uniform<i16:f32, 0.25>>) -> tensor<2x4xf32>
  func.return %out_dq : tensor<2x4xf32>
}

// The FULLY_CONNECTED blockwise filter kernel does not support a fused TANH, so
// neither the filter nor the bias is quantized.
// CHECK-LABEL: Int8SrqFullyConnectedBlockwiseFilterUnsupportedActivation
func.func @Int8SrqFullyConnectedBlockwiseFilterUnsupportedActivation(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: !tfl.blockwise_quant
  // CHECK-NOT: i32:f32
  // CHECK: %[[W_DQ:.*]] = "tfl.blockwise_dequantize"
  // CHECK: "tfl.fully_connected"(%{{.*}}, %[[W_DQ]], %{{.*}}) <{fused_activation_function = "TANH"{{.*}} : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<0.5> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 8], symmetric = false} : (tensor<4x8xi2>, tensor<4x1xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_q = "tfl.quantize"(%arg0) <{qtype = tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>}> : (tensor<2x8xf32>) -> tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>
  %in_dq = "tfl.dequantize"(%in_q) : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "TANH", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  %out_dq = "tfl.dequantize"(%out_q) : (tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>) -> tensor<2x4xf32>
  func.return %out_dq : tensor<2x4xf32>
}

// The 16-bit path of reading dense resource scales.
// CHECK-LABEL: Int8SrqFullyConnectedBlockwiseFilterF16ResourceScales
func.func @Int8SrqFullyConnectedBlockwiseFilterF16ResourceScales(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-DAG: %[[W:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense_resource<blockwise_scales_f16> : tensor<4x1xf16>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x!quant.uniform<i32:f32:0, {2.500000e-01,5.000000e-01,7.500000e-01,1.250000e-01}>>
  // CHECK: "tfl.fully_connected"(%{{.*}}, %[[W]], %[[B]])
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense_resource<blockwise_scales_f16> : tensor<4x1xf16>} : () -> tensor<4x1xf16>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 8], symmetric = false} : (tensor<4x8xi2>, tensor<4x1xf16>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_q = "tfl.quantize"(%arg0) <{qtype = tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>}> : (tensor<2x8xf32>) -> tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>
  %in_dq = "tfl.dequantize"(%in_q) : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  %out_dq = "tfl.dequantize"(%out_q) : (tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>) -> tensor<2x4xf32>
  func.return %out_dq : tensor<2x4xf32>
}

// Scales [0.5, 1.0, 1.5, 0.25] stored as dense resources.
{-#
  dialect_resources: {
    builtin: {
      blockwise_scales: "0x040000000000003F0000803F0000C03F0000803E",
      blockwise_scales_f16: "0x020000000038003C003E0034"
    }
  }
#-}
