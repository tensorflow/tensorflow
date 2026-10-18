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

// Verifier tests for the blockwise quantize/dequantize ops.
//
// These two ops exist only between `LowerQuantAnnotations` and
// `FuseA4W2DynamicRangeFullyConnected`, so they are deliberately absent from
// the TFLite schema and are not run through `-tfl-runtime-verify`.

// RUN: litert-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: testBlockwiseQuantize
func.func @testBlockwiseQuantize(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // CHECK: "tfl.blockwise_quantize"
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

func.func @testBlockwiseQuantizeDynamicShape(%arg0: tensor<?x64xf32>) -> tensor<?x64xi4> {
  // expected-error @+1 {{expects a statically shaped tensor}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true} : (tensor<?x64xf32>) -> (tensor<?x64xi4>, tensor<?x2xf8E8M0FNU>, none)
  func.return %0#0 : tensor<?x64xi4>
}

// -----

func.func @testBlockwiseQuantizeBlockShapeRank(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // expected-error @+1 {{expects block_shape of rank 2}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [32], scale_type = f8E8M0FNU, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

func.func @testBlockwiseQuantizeIndivisibleBlock(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // expected-error @+1 {{expects dimension 1 (64) to be divisible by block_shape[1] (7)}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 7], scale_type = f8E8M0FNU, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

func.func @testBlockwiseQuantizeNonPositiveBlock(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // expected-error @+1 {{expects a positive block_shape}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 0], scale_type = f8E8M0FNU, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

func.func @testBlockwiseQuantizeScaleGridMismatch(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // expected-error @+1 {{expects scale dimension 1 to be 1 or 2}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x8xf8E8M0FNU>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

func.func @testBlockwiseQuantizeScaleTypeMismatch(%arg0: tensor<4x64xf32>) -> tensor<4x64xi4> {
  // expected-error @+1 {{expects the scale element type to match scale_type}}
  %0:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf32>, none)
  func.return %0#0 : tensor<4x64xi4>
}

// -----

// CHECK-LABEL: testBlockwiseDequantize
func.func @testBlockwiseDequantize(%arg0: tensor<32x64xi2>, %arg1: tensor<32x1xf32>, %arg2: tensor<32x1xf32>) -> tensor<32x64xf32> {
  // CHECK: "tfl.blockwise_dequantize"
  %0 = "tfl.blockwise_dequantize"(%arg0, %arg1, %arg2) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

// A missing zero point is how a symmetric dequantize is spelled.
// CHECK-LABEL: testBlockwiseDequantizeNoZeroPoint
func.func @testBlockwiseDequantizeNoZeroPoint(%arg0: tensor<32x64xi2>, %arg1: tensor<32x1xf32>) -> tensor<32x64xf32> {
  %none = "tfl.no_value"() {value} : () -> none
  // CHECK: "tfl.blockwise_dequantize"
  %0 = "tfl.blockwise_dequantize"(%arg0, %arg1, %none) {block_shape = [1, 64], symmetric = true} : (tensor<32x64xi2>, tensor<32x1xf32>, none) -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

func.func @testBlockwiseDequantizeScalesRank(%arg0: tensor<32x64xi2>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
  %none = "tfl.no_value"() {value} : () -> none
  // expected-error @+1 {{expects scales of rank 2}}
  %0 = "tfl.blockwise_dequantize"(%arg0, %arg1, %none) {block_shape = [1, 64], symmetric = true} : (tensor<32x64xi2>, tensor<32xf32>, none) -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

func.func @testBlockwiseDequantizeIndivisibleBlock(%arg0: tensor<32x64xi2>, %arg1: tensor<32x1xf32>) -> tensor<32x64xf32> {
  %none = "tfl.no_value"() {value} : () -> none
  // expected-error @+1 {{expects dimension 1 (64) to be divisible by block_shape[1] (7)}}
  %0 = "tfl.blockwise_dequantize"(%arg0, %arg1, %none) {block_shape = [1, 7], symmetric = true} : (tensor<32x64xi2>, tensor<32x1xf32>, none) -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}

// -----

func.func @testBlockwiseDequantizeSymmetricWithZeroPoints(%arg0: tensor<32x64xi2>, %arg1: tensor<32x1xf32>) -> tensor<32x64xf32> {
  %zp = "tfl.pseudo_const"() {value = dense<[[0.0], [-5.000000e-01]]> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<2x64xi2>} : () -> tensor<2x64xi2>
  %s = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  // expected-error @+1 {{expects a per-tensor zero_point when symmetric is set}}
  %0 = "tfl.blockwise_dequantize"(%w, %s, %zp) {block_shape = [1, 64], symmetric = true} : (tensor<2x64xi2>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x64xf32>
  func.return %arg0 : tensor<32x64xi2>
}
