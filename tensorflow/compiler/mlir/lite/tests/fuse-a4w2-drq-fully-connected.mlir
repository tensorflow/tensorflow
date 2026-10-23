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

// RUN: litert-opt %s -tfl-fuse-a4w2-drq-fully-connected | FileCheck %s

// Every property the `cint2_fp32_int4_e8m0_drq` contract fixes is baked into the
// runtime kernel rather than carried in the flatbuffer, so each negative test
// below pins a case where matching would produce a model that runs but
// computes something other than what the graph says.

// -----------------------------------------------------------------------------
// Positive case.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @FuseA4W2Drq
func.func @FuseA4W2Drq(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // The blockwise ops collapse away entirely; the weights become a per-axis
  // i2 qconst and the float activation feeds the fully_connected directly.
  // CHECK-NOT: tfl.blockwise_quantize
  // CHECK-NOT: tfl.blockwise_dequantize
  // CHECK: %[[QW:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x64x!quant.uniform<i2:f32:0, {2.500000e-01
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[QW]],
  // CHECK-SAME: tfl.quant_spec = {act_dilation = 1.500000e+00 : f32, spec = "cint2_fp32_int4_e8m0_drq"}
  // CHECK: return %[[FC]]
}

// -----------------------------------------------------------------------------
// Negative: the weights are on a plain symmetric grid, not a centered one.
//
// The kernel unconditionally reconstructs weights as `(q + 0.5) * scale`, so
// matching here would shift every weight by half a step.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseSymmetricWeights
func.func @NoFuseSymmetricWeights(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %none) {block_shape = [1, 64], symmetric = true} : (tensor<32x64xi2>, tensor<32x1xf32>, none) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_dequantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: an asymmetric weight zero point that is not the centered -0.5.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseNonCenteredZeroPoint
func.func @NoFuseNonCenteredZeroPoint(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_dequantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: an f32 activation scale.
//
// The kernel rounds the scale up to a power of two, so an unconstrained f32
// scale would quantize the activations to different values.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseFloat32ActScale
func.func @NoFuseFloat32ActScale(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f32, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf32>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf32>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_quantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: 8 bit activations.
//
// The spec is a4w2; the kernel clamps to the 4 bit range regardless.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseInt8Activations
func.func @NoFuseInt8Activations(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi8>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi8>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_quantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: an activation block size other than 32.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseWrongActBlockSize
func.func @NoFuseWrongActBlockSize(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 16], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x4xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 16], symmetric = true} : (tensor<4x64xi4>, tensor<4x4xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_quantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: sub-channel weights, i.e. more than one block along the
// contracting axis. The collapsed op can only express one scale per channel.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseSubChannelWeights
func.func @NoFuseSubChannelWeights(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x2xf32>} : () -> tensor<32x2xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x2xf32>} : () -> tensor<32x2xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 32], symmetric = false} : (tensor<32x64xi2>, tensor<32x2xf32>, tensor<32x2xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_dequantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: asymmetric activations. The collapsed op carries no activation
// zero point.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseAsymmetricActivations
func.func @NoFuseAsymmetricActivations(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = false, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, tensor<1x1xi4>)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = false} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, tensor<1x1xi4>) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_quantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: shuffled-weights fully_connected, which has two results. The
// rewrite builds a single-result op, so replacing it would be invalid.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseShuffledWeights
func.func @NoFuseShuffledWeights(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %act#1, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0:2 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "SHUFFLED4x16INT8"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> (tensor<4x32xf32>, tensor<4x32xf32>)
  func.return %0#0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_dequantize
  // CHECK-NOT: tfl.quant_spec
}

// -----------------------------------------------------------------------------
// Negative: the dequantize consumes a scale tensor that is not the one the
// matched quantize produced. Every parameter of the fused op is read off the
// quantize, so folding this away would silently change which scale is applied.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @NoFuseMismatchedActivationScale
func.func @NoFuseMismatchedActivationScale(%arg0: tensor<4x64xf32>) -> tensor<4x32xf32> {
  %bias = "tfl.no_value"() {value} : () -> none
  %w = "tfl.pseudo_const"() {value = dense<1> : tensor<32x64xi2>} : () -> tensor<32x64xi2>
  %w_scale = "tfl.pseudo_const"() {value = dense<2.500000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %w_zp = "tfl.pseudo_const"() {value = dense<-5.000000e-01> : tensor<32x1xf32>} : () -> tensor<32x1xf32>
  %other_scale = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<4x2xf8E8M0FNU>} : () -> tensor<4x2xf8E8M0FNU>
  %act:3 = "tfl.blockwise_quantize"(%arg0) {block_shape = [1, 32], scale_type = f8E8M0FNU, symmetric = true, range_dilation = 1.500000e+00 : f32} : (tensor<4x64xf32>) -> (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none)
  %act_dq = "tfl.blockwise_dequantize"(%act#0, %other_scale, %act#2) {block_shape = [1, 32], symmetric = true} : (tensor<4x64xi4>, tensor<4x2xf8E8M0FNU>, none) -> tensor<4x64xf32>
  %w_dq = "tfl.blockwise_dequantize"(%w, %w_scale, %w_zp) {block_shape = [1, 64], symmetric = false} : (tensor<32x64xi2>, tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x64xf32>
  %0 = "tfl.fully_connected"(%act_dq, %w_dq, %bias) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  func.return %0 : tensor<4x32xf32>

  // CHECK: tfl.blockwise_dequantize
  // CHECK-NOT: tfl.quant_spec
}
