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
// RUN: litert-opt %s -tfl-bias-quantizer | FileCheck %s

// CHECK-LABEL: QuantizeConv2DPerChannel
func.func @QuantizeConv2DPerChannel(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = arith.constant dense<1.0> : tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}> {propagated}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2DPerChannelConst
func.func @QuantizeConv2DPerChannelConst(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<32xf32>} : () -> tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}> {propagated}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2DPerChannels
func.func @QuantizeConv2DPerChannels(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32:3, {1.0,2.0,3.0}>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = arith.constant dense<1.0> : tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.000000e+00,4.000000e+00,9.000000e+00}>>}> {propagated}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>} : () -> tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x12xf32>
  %5 = "tfl.fully_connected"(%2, %4, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.fully_connected"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeDepthwiseConv2D
func.func @QuantizeDepthwiseConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.depthwise_conv_2d"(%2, %4, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.depthwise_conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeSharedBiases
func.func @QuantizeSharedBiases(
    %arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x224x224x3xf32>
  %2 = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<32x3x3x3xf32>
  %conv1 = "tfl.conv_2d"(%1, %2, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %3 = "tfl.quantize"(%conv1) {qtype = tensor<1x112x112x32xf32>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>

  %4 = "tfl.dequantize"(%3) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %5 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%4, %5, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %6 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

  func.return %6 : tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>

// CHECK-DAG: %[[Q1:.*]] = "tfl.quantize"(%[[CST]])
// CHECK-DAG: %[[DQ1:.*]] = "tfl.dequantize"(%[[Q1]]) : (tensor<32x!quant.uniform<i32:f32, 1.000000e+00>>)
// CHECK-DAG: %[[Q2:.*]] = "tfl.quantize"(%[[CST]])
// CHECK-DAG: %[[DQ2:.*]] = "tfl.dequantize"(%[[Q2]]) : (tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>)

// CHECK-DAG: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[DQ1]])
// CHECK-DAG: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[DQ2]])
}

// -----

// CHECK-LABEL: QuantizeSharedBiases2
func.func @QuantizeSharedBiases2(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %1 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %add = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %3 = "tfl.quantize"(%add) {qtype = tensor<32xf32>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  %5 = "tfl.dequantize"(%arg1) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%5, %6, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %7 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>
  func.return %3, %7 : tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[Q_BIAS:.*]] = "tfl.quantize"(%[[CST]]) <{qtype = tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>}> {propagated} : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>
// CHECK: %[[DQ_BIAS:.*]] = "tfl.dequantize"(%[[Q_BIAS]])

// CHECK-DAG: tfl.add {{.*}}, %[[CST]]
// CHECK-DAG: "tfl.conv_2d"({{.*}}, {{.*}}, %[[DQ_BIAS]])

}


// -----

// Make sure biases are not shared.
// CHECK-LABEL: QuantizeSharedBiases3
func.func @QuantizeSharedBiases3(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %5 = "tfl.dequantize"(%arg1) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%5, %6, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %7 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

  %1 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %add = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %3 = "tfl.quantize"(%add) {qtype = tensor<32xf32>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %3, %7 : tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[Q_BIAS:.*]] = "tfl.quantize"(%[[CST]]) <{qtype = tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>}>
// CHECK: %[[DQ_BIAS:.*]] = "tfl.dequantize"(%[[Q_BIAS]])
// CHECK-DAG: "tfl.conv_2d"({{.*}}, {{.*}}, %[[DQ_BIAS]])
// CHECK-DAG: tfl.add {{.*}}, %[[CST]]
}

// -----

// The bias of an SRQ FULLY_CONNECTED with a blockwise filter is quantized per
// output channel with the input scale times the largest block scale of each
// channel.
// CHECK-LABEL: QuantizeFullyConnectedBlockwiseFilterBias
func.func @QuantizeFullyConnectedBlockwiseFilterBias(%arg0: tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>> {
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<[[0.5, 0.25], [1.0, 2.0], [1.5, 1.5], [0.25, 0.125]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 4], symmetric = false} : (tensor<4x8xi2>, tensor<4x2xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_dq = "tfl.dequantize"(%arg0) : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  func.return %out_q : tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>

// CHECK: %[[Q_BIAS:.*]] = "tfl.quantize"(%{{.*}}) <{qtype = tensor<4x!quant.uniform<i32:f32:0, {2.500000e-01,1.000000e+00,7.500000e-01,1.250000e-01}>>}>
// CHECK: %[[DQ_BIAS:.*]] = "tfl.dequantize"(%[[Q_BIAS]])
// CHECK: "tfl.fully_connected"({{.*}}, {{.*}}, %[[DQ_BIAS]])
}

// -----

// A float input (weight-only) leaves the bias of a blockwise filter float.
// CHECK-LABEL: NotQuantizeWeightOnlyFullyConnectedBlockwiseFilterBias
func.func @NotQuantizeWeightOnlyFullyConnectedBlockwiseFilterBias(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: "tfl.quantize"
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<0.5> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %none) {block_shape = [1, 8], symmetric = true} : (tensor<4x8xi2>, tensor<4x1xf32>, none) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %fc = "tfl.fully_connected"(%arg0, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  func.return %fc : tensor<2x4xf32>
}

// -----

// A unit block of 2 shares each scale between two output channels.
// CHECK-LABEL: QuantizeFullyConnectedUnitBlockedBlockwiseFilterBias
func.func @QuantizeFullyConnectedUnitBlockedBlockwiseFilterBias(%arg0: tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>> {
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<[[0.5], [2.0]]> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [2, 8], symmetric = false} : (tensor<4x8xi2>, tensor<2x1xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %bias = arith.constant dense<1.0> : tensor<4xf32>
  %in_dq = "tfl.dequantize"(%arg0) : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %bias) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  func.return %out_q : tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>

// CHECK: "tfl.quantize"(%{{.*}}) <{qtype = tensor<4x!quant.uniform<i32:f32:0, {2.500000e-01,2.500000e-01,1.000000e+00,1.000000e+00}>>}>
}

// -----

// The FULLY_CONNECTED blockwise filter kernel needs int8 input and output, so
// the filter of a 16x8 op is not folded and its bias stays float.
// CHECK-LABEL: NotQuantizeInt16FullyConnectedBlockwiseFilterBias
func.func @NotQuantizeInt16FullyConnectedBlockwiseFilterBias(%arg0: tensor<2x8x!quant.uniform<i16:f32, 0.5>>) -> tensor<2x4x!quant.uniform<i16:f32, 0.25>> {
  // CHECK-NOT: i32:f32
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<4x8xi2>} : () -> tensor<4x8xi2>
  %scales = "tfl.pseudo_const"() {value = dense<0.5> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 8], symmetric = false} : (tensor<4x8xi2>, tensor<4x1xf32>, tensor<1x1xf32>) -> tensor<4x8xf32>
  %cst = arith.constant dense<1.0> : tensor<4xf32>
  %in_dq = "tfl.dequantize"(%arg0) : (tensor<2x8x!quant.uniform<i16:f32, 0.5>>) -> tensor<2x8xf32>
  %fc = "tfl.fully_connected"(%in_dq, %w_dq, %cst) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x8xf32>, tensor<4x8xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%fc) <{qtype = tensor<2x4x!quant.uniform<i16:f32, 0.25>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i16:f32, 0.25>>
  func.return %out_q : tensor<2x4x!quant.uniform<i16:f32, 0.25>>
}
