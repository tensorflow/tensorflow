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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// Round trips `!tfl.blockwise_quant` constants through
// `BlockwiseQuantization.block_shape`. Integer zero points are serialized as
// int32 tensors.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<2x4xf32>) -> (tensor<4x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  // Grouped channels: 2 adjacent rows share each scale.
  // CHECK: %[[W0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>, value = dense<{{\[}}[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [-5, -6, -7, -8]]> : tensor<4x4xi8>}>
  %w0 = "tfl.pseudo_qconst"() {qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, value = dense<[[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [-5, -6, -7, -8]]> : tensor<4x4xi8>} : () -> tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>
  // CHECK: "tfl.dequantize"(%[[W0]])
  %0 = "tfl.dequantize"(%w0) : (tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>) -> tensor<4x4xf32>

  // Per-channel on axis 0 with blocks of 2 along axis 1, i4 storage and a
  // broadcast zero point grid.
  // CHECK: %[[W1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>, zero_points = dense<{{\[}}[1, -1]]> : tensor<1x2xi32>>>, value = dense<{{\[}}[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>}>
  %w1 = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>, value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>
  // CHECK: "tfl.dequantize"(%[[W1]])
  %1 = "tfl.dequantize"(%w1) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>) -> tensor<2x4xf32>

  // f16 scales and a float zero point.
  // CHECK: %[[W2:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf16>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>
  %w2 = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<1> : tensor<2x4xi8>} : () -> tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  // CHECK: "tfl.dequantize"(%[[W2]])
  %2 = "tfl.dequantize"(%w2) : (tensor<2x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32>

  func.return %0, %1, %2 : tensor<4x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}
