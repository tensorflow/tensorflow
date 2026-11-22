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

// RUN: litert-opt %s -split-input-file -tfl-lower-blockwise-dequantize | FileCheck %s

// Grouped channels on weights: two adjacent output channels share a scale.
// CHECK-LABEL: @groupedChannelWeights
func.func @groupedChannelWeights(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %w = "tfl.pseudo_const"() {value = dense<[[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [-5, -6, -7, -8]]> : tensor<4x4xi8>} : () -> tensor<4x4xi8>
  %scales = "tfl.pseudo_const"() {value = dense<[[0.5], [0.25]]> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %none) {block_shape = [2, 4], symmetric = true} : (tensor<4x4xi8>, tensor<2x1xf32>, none) -> tensor<4x4xf32>
  %0 = "tfl.fully_connected"(%arg0, %w_dq, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x4xf32>, tensor<4x4xf32>, none) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>

  // CHECK-NOT: tfl.blockwise_dequantize
  // CHECK:     %[[Q:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>, value = dense<{{\[}}[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [-5, -6, -7, -8]]> : tensor<4x4xi8>}>
  // CHECK:     %[[DQ:.*]] = "tfl.dequantize"(%[[Q]]) : (tensor<4x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<4x4xf32>
  // CHECK:     "tfl.fully_connected"(%arg0, %[[DQ]], %{{.*}})
}

// -----

// Per-channel on axis 0 with blocks along axis 1, i4 storage and float zero
// points.
// CHECK-LABEL: @perChannelAndBlockedWithZeroPoints
func.func @perChannelAndBlockedWithZeroPoints() -> tensor<2x4xf32> {
  %w = "tfl.pseudo_const"() {value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4xi4>
  %scales = "tfl.pseudo_const"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %zp = "tfl.pseudo_const"() {value = dense<-0.5> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %zp) {block_shape = [1, 2], symmetric = false} : (tensor<2x4xi4>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  func.return %w_dq : tensor<2x4xf32>

  // CHECK: "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<{{.*}}> : tensor<2x2xf32>, zero_points = dense<-5.000000e-01> : tensor<2x2xf32>>>
  // CHECK: "tfl.dequantize"
}

// -----

// Dynamically quantized activations are left alone: their scales are only
// known at runtime.
// CHECK-LABEL: @dynamicActivationsAreNotLowered
func.func @dynamicActivationsAreNotLowered(%arg0: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f32, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf32>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf32>, none) -> tensor<4x64xf32>
  func.return %act_dq : tensor<4x64xf32>

  // CHECK: "tfl.blockwise_dequantize"
  // CHECK-NOT: "tfl.dequantize"
}

// -----

// A float16 output: `tfl.dequantize` to f32, then `tfl.cast`.
// CHECK-LABEL: @float16Output
func.func @float16Output() -> tensor<2x4xf16> {
  %w = "tfl.pseudo_const"() {value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4xi4>
  %scales = "tfl.pseudo_const"() {value = dense<[[1.0], [2.0]]> : tensor<2x1xf16>} : () -> tensor<2x1xf16>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %none) {block_shape = [1, 4], symmetric = true} : (tensor<2x4xi4>, tensor<2x1xf16>, none) -> tensor<2x4xf16>
  func.return %w_dq : tensor<2x4xf16>

  // CHECK-NOT: tfl.blockwise_dequantize
  // CHECK:     %[[Q:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{.*}}> : tensor<2x1xf16>>>
  // CHECK:     %[[DQ:.*]] = "tfl.dequantize"(%[[Q]]) : (tensor<2x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32>
  // CHECK:     %[[CAST:.*]] = "tfl.cast"(%[[DQ]]) : (tensor<2x4xf32>) -> tensor<2x4xf16>
  // CHECK:     return %[[CAST]]
}

// -----

// A bfloat16 output: `tfl.dequantize` to f32, then `tfl.cast`.
// CHECK-LABEL: @bfloat16Output
func.func @bfloat16Output() -> tensor<2x4xbf16> {
  %w = "tfl.pseudo_const"() {value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4xi4>
  %scales = "tfl.pseudo_const"() {value = dense<[[1.0], [2.0]]> : tensor<2x1xbf16>} : () -> tensor<2x1xbf16>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %none) {block_shape = [1, 4], symmetric = true} : (tensor<2x4xi4>, tensor<2x1xbf16>, none) -> tensor<2x4xbf16>
  func.return %w_dq : tensor<2x4xbf16>

  // CHECK-NOT: tfl.blockwise_dequantize
  // CHECK:     %[[Q:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{.*}}> : tensor<2x1xbf16>>>
  // CHECK:     %[[DQ:.*]] = "tfl.dequantize"(%[[Q]]) : (tensor<2x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32>
  // CHECK:     %[[CAST:.*]] = "tfl.cast"(%[[DQ]]) : (tensor<2x4xf32>) -> tensor<2x4xbf16>
  // CHECK:     return %[[CAST]]
}
