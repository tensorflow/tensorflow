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

// Parser, printer and verifier tests for `!tfl.blockwise_quant`.

// RUN: litert-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// Grouped-channel weights: 2 adjacent output channels share each scale.
// CHECK-LABEL: groupedChannel
func.func @groupedChannel(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: "tfl.pseudo_qconst"() <{qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>
  // CHECK-SAME: -> tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>
  %w = "tfl.pseudo_qconst"() {qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, value = dense<1> : tensor<4x4xi8>} : () -> tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>
  // CHECK: "tfl.dequantize"
  %dq = "tfl.dequantize"(%w) : (tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>) -> tensor<4x4xf32>
  %none = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.fully_connected"(%arg0, %dq, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x4xf32>, tensor<4x4xf32>, none) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// Per-channel on axis 0 combined with blocks of 2 along axis 1, with zero
// points and a broadcast (extent 1) zero point grid.
// CHECK-LABEL: perChannelAndBlocked
func.func @perChannelAndBlocked() -> tensor<2x4xf32> {
  // CHECK: !tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<{{.*}}> : tensor<2x2xf32>, zero_points = dense<{{.*}}> : tensor<1x2xi4>>
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>, value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>
  %dq = "tfl.dequantize"(%w) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>) -> tensor<2x4xf32>
  func.return %dq : tensor<2x4xf32>
}

// -----

// Non-f32 scales and float zero points are allowed.
// CHECK-LABEL: f16ScalesFloatZeroPoints
func.func @f16ScalesFloatZeroPoints() -> tensor<2x4xf32> {
  // CHECK: !tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<{{.*}}> : tensor<2x1xf16>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<1> : tensor<2x4xi2>} : () -> tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  %dq = "tfl.dequantize"(%w) : (tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32>
  func.return %dq : tensor<2x4xf32>
}

// -----

// Partial blocks: the grid is ceil(shape / block_shape).
// CHECK-LABEL: partialBlocks
func.func @partialBlocks() -> tensor<1x5xf32> {
  // CHECK: block_shape = [1, 2]
  %w = "tfl.pseudo_qconst"() {qtype = tensor<1x5x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<1x3xf32>>>, value = dense<1> : tensor<1x5xi8>} : () -> tensor<1x5x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<1x3xf32>>>
  %dq = "tfl.dequantize"(%w) : (tensor<1x5x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<1x3xf32>>>) -> tensor<1x5xf32>
  func.return %dq : tensor<1x5xf32>
}

// -----

func.func @scalesRankMismatch() {
  // expected-error @+1 {{expects scales of rank 2 to match block_shape}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2xf32>>>
  func.return
}

// -----

func.func @nonPositiveBlockShape() {
  // expected-error @+1 {{expects a positive block_shape}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [0, 4], scales = dense<1.0> : tensor<1x1xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [0, 4], scales = dense<1.0> : tensor<1x1xf32>>>
  func.return
}

// -----

func.func @floatStorage() {
  // expected-error @+1 {{expects an integer storage type}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<f16 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2x1xf32>>>, value = dense<1.0> : tensor<2x4xf16>} : () -> tensor<2x4x!tfl.blockwise_quant<f16 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2x1xf32>>>
  func.return
}

// -----

func.func @integerScales() {
  // expected-error @+1 {{expects floating point scales}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<1> : tensor<2x1xi32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<1> : tensor<2x1xi32>>>
  func.return
}

// -----

func.func @blockShapeRankMismatch() {
  // expected-error @+1 {{expects one block_shape entry per dimension of a tensor of rank 2}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [4], scales = dense<1.0> : tensor<1xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [4], scales = dense<1.0> : tensor<1xf32>>>
  func.return
}

// -----

func.func @scalesDoNotMatchGrid() {
  // expected-error @+1 {{expects scales dimension 1 to be 1 or 2 (the number of blocks)}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<2x3xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<2x3xf32>>>
  func.return
}

// -----

func.func @valueDoesNotMatchStorage() {
  // expected-error @+1 {{expects the value element type to match the storage type of qtype ('i4'), got 'i8'}}
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2x1xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<1.0> : tensor<2x1xf32>>>
  func.return
}

// -----

func.func @dequantizeGridMismatch(%arg0: tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<2x3xf32>>>) -> tensor<2x4xf32> {
  // expected-error @+1 {{expects scales dimension 1 to be 1 or 2 (the number of blocks)}}
  %dq = "tfl.dequantize"(%arg0) : (tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 2], scales = dense<1.0> : tensor<2x3xf32>>>) -> tensor<2x4xf32>
  func.return %dq : tensor<2x4xf32>
}
