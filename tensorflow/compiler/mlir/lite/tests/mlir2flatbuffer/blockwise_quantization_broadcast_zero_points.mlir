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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s --check-prefix=MLIR

// Axes along which blockwise zero points are uniform are collapsed to extent 1
// on export while preserving rank.

// CHECK:      tensors: [ {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK-NEXT:   type: INT2,
// CHECK:        name: "tfl.pseudo_qconst",
// CHECK:        details_type: BlockwiseQuantization,
// CHECK-NEXT:   details: {
// CHECK-NEXT:     scales: 1,
// CHECK-NEXT:     zero_points: 2,
// CHECK-NEXT:     block_shape: [ 1, 2 ]
// CHECK-NEXT:   }
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 2 ],
// CHECK:        name: "tfl.pseudo_qconst_scales",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 1, 1 ],
// CHECK:        buffer: 3,
// CHECK-NEXT:   name: "tfl.pseudo_qconst_zero_points",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK:        name: "tfl.dequantize",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK-NEXT:   type: INT4,
// CHECK:        name: "tfl.pseudo_qconst1",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 2 ],
// CHECK:        name: "tfl.pseudo_qconst1_scales",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 1 ],
// CHECK-NEXT:   type: INT32,
// CHECK-NEXT:   buffer: 7,
// CHECK-NEXT:   name: "tfl.pseudo_qconst1_zero_points",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK:        name: "tfl.dequantize1",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK-NEXT:   type: INT4,
// CHECK:        name: "tfl.pseudo_qconst2",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 2 ],
// CHECK:        name: "tfl.pseudo_qconst2_scales",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 1, 2 ],
// CHECK-NEXT:   type: INT32,
// CHECK-NEXT:   buffer: 11,
// CHECK-NEXT:   name: "tfl.pseudo_qconst2_zero_points",

// Buffer 3 holds a single f32 (-0.5f = 0xbf000000).
// CHECK:      buffers: [ {
// CHECK:      }, {
// CHECK:      }, {
// CHECK:      }, {
// CHECK-NEXT:   data: [ 0, 0, 0, 191 ]
// CHECK-NEXT: }, {

// MLIR-LABEL: func @main
// MLIR: zero_points = dense<-5.000000e-01> : tensor<1x1xf32>
// MLIR: zero_points = dense<{{\[}}[1], [-1]]> : tensor<2x1xi32>
// MLIR: zero_points = dense<{{\[}}[1, -1]]> : tensor<1x2xi32>

func.func @main() -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %w0 = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<-0.5> : tensor<2x2xf32>>>, value = dense<1> : tensor<2x4xi2>} : () -> tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<-0.5> : tensor<2x2xf32>>>
  %0 = "tfl.dequantize"(%w0) : (tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<-0.5> : tensor<2x2xf32>>>) -> tensor<2x4xf32>

  %w1 = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, 1], [-1, -1]]> : tensor<2x2xi4>>>, value = dense<1> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, 1], [-1, -1]]> : tensor<2x2xi4>>>
  %1 = "tfl.dequantize"(%w1) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, 1], [-1, -1]]> : tensor<2x2xi4>>>) -> tensor<2x4xf32>

  %w2 = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1], [1, -1]]> : tensor<2x2xi4>>>, value = dense<1> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1], [1, -1]]> : tensor<2x2xi4>>>
  %2 = "tfl.dequantize"(%w2) : (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 2], scales = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>, zero_points = dense<[[1, -1], [1, -1]]> : tensor<2x2xi4>>>) -> tensor<2x4xf32>

  func.return %0, %1, %2 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}
