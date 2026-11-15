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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// A `!tfl.blockwise_quant` constant is serialized as `BlockwiseQuantization`
// with a `block_shape`. Its scales and zero points are written to tensors of
// their own, right after the quantized tensor.

func.func @main() -> tensor<2x4xf32> {
  // CHECK:      tensors: [ {
  // CHECK-NEXT:   shape: [ 2, 4 ],
  // CHECK-NEXT:   type: INT4,
  // CHECK-NEXT:   buffer: 1,
  // CHECK-NEXT:   name: "tfl.pseudo_qconst",
  // CHECK-NEXT:   quantization: {
  // CHECK-NEXT:     details_type: BlockwiseQuantization,
  // CHECK-NEXT:     details: {
  // CHECK-NEXT:       scales: 1,
  // CHECK-NEXT:       zero_points: 2,
  // CHECK-NEXT:       block_shape: [ 1, 2 ]
  // CHECK-NEXT:     }
  // CHECK-NEXT:   },
  // CHECK-NEXT:   has_rank: true
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   shape: [ 2, 2 ],
  // CHECK-NEXT:   buffer: 2,
  // CHECK-NEXT:   name: "tfl.pseudo_qconst_scales",
  // CHECK-NEXT:   has_rank: true
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   shape: [ 1, 2 ],
  // CHECK-NEXT:   type: INT32,
  // CHECK-NEXT:   buffer: 3,
  // CHECK-NEXT:   name: "tfl.pseudo_qconst_zero_points",
  // CHECK-NEXT:   has_rank: true
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   shape: [ 2, 4 ],
  // CHECK-NEXT:   buffer: 4,
  // CHECK-NEXT:   name: "tfl.dequantize",

  // CHECK:      operators: [ {
  // CHECK-NEXT:   inputs: [ 0 ],
  // CHECK-NEXT:   outputs: [ 3 ]
  // CHECK-NEXT: } ],

  // The int4 values are packed two per byte, the scales are f32 and the zero
  // points int32.
  // CHECK:      buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   data: [ 33, 67, 239, 205 ]
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64 ]
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   data: [ 1, 0, 0, 0, 255, 255, 255, 255 ]
  // CHECK-NEXT: }, {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>, value = dense<[[1, 2, 3, 4], [-1, -2, -3, -4]]> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>
  %0 = "tfl.dequantize"(%w) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1]]> : tensor<1x2xi4>>>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}
