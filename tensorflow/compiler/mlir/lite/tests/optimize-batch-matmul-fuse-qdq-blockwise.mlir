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
// RUN: litert-opt %s -tfl-optimize-batch-matmul -tfl-large-constant-fold -tfl-bias-quantizer -tfl-fuse-qdq | FileCheck %s

// Checks the full path of an int8 SRQ batch_matmul with a single-use rank-2
// blockwise-dequantized RHS: it becomes a fully_connected, the filter transpose
// is pushed into the blockwise_dequantize constants, and the transposed filter
// is folded into a blockwise quantized FULLY_CONNECTED filter.

// CHECK-LABEL: SrqBatchmatmulBlockwiseRhsToBlockwiseFullyConnected
func.func @SrqBatchmatmulBlockwiseRhsToBlockwiseFullyConnected(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: "tfl.batch_matmul"
  // CHECK-NOT: "tfl.transpose"
  // CHECK-NOT: "tfl.blockwise_dequantize"
  // CHECK-DAG: %[[Q_IN:.*]] = "tfl.quantize"(%arg0)
  // CHECK-DAG: %[[Q_W:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<{{\[}}[5.000000e-01], [1.000000e+00], [1.500000e+00], [2.500000e-01]]> : tensor<4x1xf32>>>, value = dense<{{\[}}[-2, 1, 0, -1, -2, 1, 0, -1], [-1, 0, 1, -2, -1, 0, 1, -2], [0, -1, -2, 1, 0, -1, -2, 1], [1, -2, -1, 0, 1, -2, -1, 0]]> : tensor<4x8xi2>}>
  // CHECK: "tfl.fully_connected"(%[[Q_IN]], %[[Q_W]], %{{.*}}) {{.*}} -> tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01:1>>
  // CHECK-NOT: "tfl.batch_matmul"
  %w = arith.constant dense<[[-2, -1, 0, 1], [1, 0, -1, -2], [0, 1, -2, -1], [-1, -2, 1, 0], [-2, -1, 0, 1], [1, 0, -1, -2], [0, 1, -2, -1], [-1, -2, 1, 0]]> : tensor<8x4xi2>
  %scales = arith.constant dense<[[0.5, 1.0, 1.5, 0.25]]> : tensor<1x4xf32>
  %none = "tfl.no_value"() {value} : () -> none
  %w_dq = "tfl.blockwise_dequantize"(%w, %scales, %none) {block_shape = [8, 1], symmetric = true} : (tensor<8x4xi2>, tensor<1x4xf32>, none) -> tensor<8x4xf32>
  %q = "tfl.quantize"(%arg0) <{qtype = tensor<2x8x!quant.uniform<i8:f32, 5.000000e-01:-2>>}> : (tensor<2x8xf32>) -> tensor<2x8x!quant.uniform<i8:f32, 5.000000e-01:-2>>
  %dq = "tfl.dequantize"(%q) : (tensor<2x8x!quant.uniform<i8:f32, 5.000000e-01:-2>>) -> tensor<2x8xf32>
  %bmm = "tfl.batch_matmul"(%dq, %w_dq) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
  %out_q = "tfl.quantize"(%bmm) <{qtype = tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01:1>>}> : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01:1>>
  %out_dq = "tfl.dequantize"(%out_q) : (tensor<2x4x!quant.uniform<i8:f32, 2.500000e-01:1>>) -> tensor<2x4xf32>
  func.return %out_dq : tensor<2x4xf32>
}
