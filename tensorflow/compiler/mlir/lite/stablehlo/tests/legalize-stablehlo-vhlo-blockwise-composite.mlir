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
// RUN: odml-to-stablehlo-opt %s --stablehlo-legalize-vhlo -reconcile-unrealized-casts -split-input-file | FileCheck %s

// Direct `tfl.dequantize` of a `!tfl.blockwise_quant` constant is sunk across
// `vhlo.composite_v1` into the decomposition function.

// CHECK-LABEL: func.func @sink_blockwise_dequantize(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK:         %[[W:.*]] = "tfl.pseudo_qconst"()
// CHECK-NOT:     tfl.dequantize
// CHECK:         %[[RES:.*]] = "vhlo.composite_v1"(%[[ARG0]], %[[W]])
// CHECK-SAME:      decomposition = #vhlo.string_v1<"test.blockwise.impl">
// CHECK-SAME:      name = #vhlo.string_v1<"odml.test_blockwise">
// CHECK-SAME:      : (tensor<2x4xf32>, tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{\[\[}}5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>) -> tensor<2x4xf32>
// CHECK:         return %[[RES]] : tensor<2x4xf32>
// CHECK:       }
// CHECK-LABEL: func.func private @test.blockwise.impl(
// CHECK-SAME:    %[[D_ARG0:.*]]: tensor<2x4xf32>,
// CHECK-SAME:    %[[D_ARG1:.*]]: tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{\[\[}}5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>) -> tensor<2x4xf32> {
// CHECK:         %[[DQ:.*]] = "tfl.dequantize"(%[[D_ARG1]]) : (tensor<2x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32>
// CHECK:         %[[ADD:.*]] = tfl.add %[[D_ARG0]], %[[DQ]] {fused_activation_function = "NONE"} : tensor<2x4xf32>
// CHECK:         return %[[ADD]] : tensor<2x4xf32>
// CHECK:       }
func.func @sink_blockwise_dequantize(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>
  %dq = "tfl.dequantize"(%w) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>) -> tensor<2x4xf32>
  %0 = stablehlo.composite "odml.test_blockwise" %arg0, %dq {decomposition = @test.blockwise.impl} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
func.func private @test.blockwise.impl(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// `tfl.dequantize` followed by `tfl.cast` (e.g. f32 -> f16) of a
// `!tfl.blockwise_quant` constant is sunk across `vhlo.composite_v1`, inserting
// both `tfl.dequantize` and `tfl.cast` at the entry of the decomposition.

// CHECK-LABEL: func.func @sink_blockwise_dequantize_and_cast(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x4xf16>) -> tensor<2x4xf16> {
// CHECK:         %[[W:.*]] = "tfl.pseudo_qconst"()
// CHECK-NOT:     tfl.dequantize
// CHECK-NOT:     tfl.cast
// CHECK:         %[[RES:.*]] = "vhlo.composite_v1"(%[[ARG0]], %[[W]])
// CHECK-SAME:      decomposition = #vhlo.string_v1<"test.blockwise_cast.impl">
// CHECK-SAME:      : (tensor<2x4xf16>, tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<{{\[\[}}5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>) -> tensor<2x4xf16>
// CHECK:         return %[[RES]] : tensor<2x4xf16>
// CHECK:       }
// CHECK-LABEL: func.func private @test.blockwise_cast.impl(
// CHECK-SAME:    %[[D_ARG0:.*]]: tensor<2x4xf16>,
// CHECK-SAME:    %[[D_ARG1:.*]]: tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<{{\[\[}}5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>) -> tensor<2x4xf16> {
// CHECK:         %[[DQ:.*]] = "tfl.dequantize"(%[[D_ARG1]]) : (tensor<2x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32>
// CHECK:         %[[CAST:.*]] = "tfl.cast"(%[[DQ]]) : (tensor<2x4xf32>) -> tensor<2x4xf16>
// CHECK:         %[[ADD:.*]] = tfl.add %[[D_ARG0]], %[[CAST]] {fused_activation_function = "NONE"} : tensor<2x4xf16>
// CHECK:         return %[[ADD]] : tensor<2x4xf16>
// CHECK:       }
func.func @sink_blockwise_dequantize_and_cast(%arg0: tensor<2x4xf16>) -> tensor<2x4xf16> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<1> : tensor<2x4xi2>} : () -> tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  %dq = "tfl.dequantize"(%w) : (tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32>
  %cast = "tfl.cast"(%dq) : (tensor<2x4xf32>) -> tensor<2x4xf16>
  %0 = stablehlo.composite "odml.test_blockwise_cast" %arg0, %cast {decomposition = @test.blockwise_cast.impl} : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}
func.func private @test.blockwise_cast.impl(%arg0: tensor<2x4xf16>, %arg1: tensor<2x4xf16>) -> tensor<2x4xf16> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}

// -----

// Non-blockwise `tfl.dequantize` (e.g. uniform quantized) is left untouched.

// CHECK-LABEL: func.func @uniform_dequantize_not_sunk(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK:         %[[W:.*]] = "tfl.pseudo_qconst"()
// CHECK:         %[[DQ:.*]] = "tfl.dequantize"(%[[W]]) : (tensor<2x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x4xf32>
// CHECK:         %[[RES:.*]] = "vhlo.composite_v1"(%[[ARG0]], %[[DQ]])
// CHECK-SAME:      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:         return %[[RES]] : tensor<2x4xf32>
// CHECK:       }
func.func @uniform_dequantize_not_sunk(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!quant.uniform<i8:f32, 0.5>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!quant.uniform<i8:f32, 0.5>>
  %dq = "tfl.dequantize"(%w) : (tensor<2x4x!quant.uniform<i8:f32, 0.5>>) -> tensor<2x4xf32>
  %0 = stablehlo.composite "odml.test_uniform" %arg0, %dq {decomposition = @test.uniform.impl} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
func.func private @test.uniform.impl(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// SRQ `tfl.fully_connected` and `tfl.batch_matmul` in the decomposition
// function fuse the sunk `!tfl.blockwise_quant` weight and activation Q/DQ.

// CHECK-LABEL: func.func @fuse_srq_fc_and_bmm_in_decomposition(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK-DAG:     %[[FC_W:.*]] = "tfl.pseudo_qconst"()
// CHECK-DAG:     %[[BMM_W:.*]] = "tfl.pseudo_qconst"()
// CHECK-NOT:     tfl.dequantize
// CHECK:         %[[RES:.*]] = "vhlo.composite_v1"(%[[ARG0]], %[[FC_W]], %[[BMM_W]])
// CHECK:         return %[[RES]] : tensor<2x4xf32>
// CHECK:       }
// CHECK-LABEL: func.func private @test.srq.impl(
// CHECK-SAME:    %[[D_X:.*]]: tensor<2x4xf32>,
// CHECK-SAME:    %[[D_FC_W:.*]]: tensor<4x4x!tfl.blockwise_quant<{{.*}}>>,
// CHECK-SAME:    %[[D_BMM_W:.*]]: tensor<4x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32> {
// CHECK:         %[[NONE:.*]] = "tfl.no_value"()
// CHECK:         %[[Q_IN:.*]] = "tfl.quantize"(%[[D_X]])
// CHECK-NOT:     tfl.dequantize
// CHECK:         %[[FC:.*]] = "tfl.fully_connected"(%[[Q_IN]], %[[D_FC_W]], %[[NONE]]) <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<2x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<4x4x!tfl.blockwise_quant<{{.*}}>>, none) -> tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK-NOT:     tfl.dequantize
// CHECK:         %[[BMM:.*]] = "tfl.batch_matmul"(%[[FC]], %[[D_BMM_W]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<4x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.250000e-01>>
// CHECK:         %[[DQ_OUT:.*]] = "tfl.dequantize"(%[[BMM]]) : (tensor<2x4x!quant.uniform<i8:f32, 1.250000e-01>>) -> tensor<2x4xf32>
// CHECK:         return %[[DQ_OUT]] : tensor<2x4xf32>
// CHECK:       }
func.func @fuse_srq_fc_and_bmm_in_decomposition(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %fc_w = "tfl.pseudo_qconst"() {qtype = tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<0.5> : tensor<4x1xf32>>>, value = dense<1> : tensor<4x4xi2>} : () -> tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<0.5> : tensor<4x1xf32>>>
  %fc_dq = "tfl.dequantize"(%fc_w) : (tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<0.5> : tensor<4x1xf32>>>) -> tensor<4x4xf32>
  %bmm_w = "tfl.pseudo_qconst"() {qtype = tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [4, 1], scales = dense<0.5> : tensor<1x4xf32>>>, value = dense<1> : tensor<4x4xi2>} : () -> tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [4, 1], scales = dense<0.5> : tensor<1x4xf32>>>
  %bmm_dq = "tfl.dequantize"(%bmm_w) : (tensor<4x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [4, 1], scales = dense<0.5> : tensor<1x4xf32>>>) -> tensor<4x4xf32>
  %0 = stablehlo.composite "odml.test_srq" %arg0, %fc_dq, %bmm_dq {decomposition = @test.srq.impl} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
func.func private @test.srq.impl(%arg0: tensor<2x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<2x4xf32> {
  %none = "tfl.no_value"() {value} : () -> none
  %q_in = "tfl.quantize"(%arg0) {qtype = tensor<2x4x!quant.uniform<i8:f32, 0.5>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.5>>
  %dq_in = "tfl.dequantize"(%q_in) : (tensor<2x4x!quant.uniform<i8:f32, 0.5>>) -> tensor<2x4xf32>
  %fc = "tfl.fully_connected"(%dq_in, %arg1, %none) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<2x4xf32>, tensor<4x4xf32>, none) -> tensor<2x4xf32>
  %q_mid = "tfl.quantize"(%fc) {qtype = tensor<2x4x!quant.uniform<i8:f32, 0.25>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25>>
  %dq_mid = "tfl.dequantize"(%q_mid) : (tensor<2x4x!quant.uniform<i8:f32, 0.25>>) -> tensor<2x4xf32>
  %bmm = "tfl.batch_matmul"(%dq_mid, %arg2) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
  %q_out = "tfl.quantize"(%bmm) {qtype = tensor<2x4x!quant.uniform<i8:f32, 0.125>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 0.125>>
  %dq_out = "tfl.dequantize"(%q_out) : (tensor<2x4x!quant.uniform<i8:f32, 0.125>>) -> tensor<2x4xf32>
  return %dq_out : tensor<2x4xf32>
}
