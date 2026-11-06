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

// CHECK:      operator_codes: [ {
// CHECK-NEXT:   deprecated_builtin_code: 7,
// CHECK-NEXT:   version: 6,
// CHECK-NEXT:   builtin_code: EMBEDDING_LOOKUP
// CHECK-NEXT: } ],

// CHECK:      tensors: [ {
// CHECK-NEXT:   shape: [ 2 ],
// CHECK-NEXT:   type: INT32,
// CHECK:        name: "arg0",
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 4, 8 ],
// CHECK-NEXT:   type: INT2,
// CHECK-NEXT:   buffer: 2,
// CHECK-NEXT:   name: "tfl.pseudo_qconst",
// CHECK-NEXT:   quantization: {
// CHECK-NEXT:     details_type: BlockwiseQuantization,
// CHECK-NEXT:     details: {
// CHECK-NEXT:       scales: 2,
// CHECK-NEXT:       zero_points: 3,
// CHECK-NEXT:       block_shape: [ 1, 8 ]
// CHECK-NEXT:     }
// CHECK-NEXT:   },
// CHECK-NEXT:   has_rank: true
// CHECK-NEXT: }, {
// CHECK-NEXT:   shape: [ 4, 1 ],
// CHECK-NEXT:   type: FLOAT16,
// CHECK-NEXT:   buffer: 3,
// CHECK-NEXT:   name: "tfl.pseudo_qconst_scales",
// CHECK-NEXT:   has_rank: true
// CHECK-NEXT: }, {
// CHECK-NEXT:   shape: [ 1, 1 ],
// CHECK-NEXT:   buffer: 4,
// CHECK-NEXT:   name: "tfl.pseudo_qconst_zero_points",
// CHECK-NEXT:   has_rank: true
// CHECK-NEXT: }, {
// CHECK-NEXT:   shape: [ 2, 8 ],
// CHECK-NEXT:   buffer: 5,
// CHECK-NEXT:   name: "tfl.embedding_lookup",

// MLIR-LABEL: func.func @main
// MLIR: %[[TABLE:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<{{.*}}> : tensor<4x1xf16>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>
// MLIR: "tfl.embedding_lookup"(%arg0, %[[TABLE]]) : (tensor<2xi32>, tensor<4x8x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x8xf32>

func.func @main(%arg0: tensor<2xi32>) -> tensor<2x8xf32> {
  %table = "tfl.pseudo_qconst"() {qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<[[-2, -1, 0, 1, -2, -1, 0, 1], [1, 0, -1, -2, 1, 0, -1, -2], [0, 1, -2, -1, 0, 1, -2, -1], [-1, -2, 1, 0, -1, -2, 1, 0]]> : tensor<4x8xi2>} : () -> tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  %out = "tfl.embedding_lookup"(%arg0, %table) : (tensor<2xi32>, tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x8xf32>
  func.return %out : tensor<2x8xf32>
}
