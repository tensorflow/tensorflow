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
// CHECK-NEXT:   deprecated_builtin_code: 9,
// CHECK-NEXT:   version: 16,
// CHECK-NEXT:   builtin_code: FULLY_CONNECTED
// CHECK-NEXT: } ],

// CHECK:      tensors: [ {
// CHECK-NEXT:   shape: [ 2, 8 ],
// CHECK-NEXT:   type: INT8,
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
// CHECK-NEXT:   shape: [ 4 ],
// CHECK-NEXT:   type: INT32,
// CHECK:        quantization: {
// CHECK-NEXT:     scale: [ 0.25, 0.5, 0.75, 0.125 ],
// CHECK-NEXT:     zero_point: [ 0, 0, 0, 0 ]
// CHECK-NEXT:   },
// CHECK:      }, {
// CHECK-NEXT:   shape: [ 2, 4 ],
// CHECK-NEXT:   type: INT8,

// MLIR-LABEL: func.func @main
// MLIR: %[[FILTER:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<{{.*}}> : tensor<4x1xf16>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>
// MLIR: "tfl.fully_connected"(%arg0, %[[FILTER]], %{{.*}})

func.func @main(%arg0: tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>> {
  %filter = "tfl.pseudo_qconst"() {qtype = tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<[[-2, -1, 0, 1, -2, -1, 0, 1], [1, 0, -1, -2, 1, 0, -1, -2], [0, 1, -2, -1, 0, 1, -2, -1], [-1, -2, 1, 0, -1, -2, 1, 0]]> : tensor<4x8xi2>} : () -> tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  %bias = "tfl.pseudo_qconst"() {qtype = tensor<4x!quant.uniform<i32:f32:0, {0.25, 0.5, 0.75, 0.125}>>, value = dense<[1, -2, 3, -4]> : tensor<4xi32>} : () -> tensor<4x!quant.uniform<i32:f32:0, {0.25, 0.5, 0.75, 0.125}>>
  %out = "tfl.fully_connected"(%arg0, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x8x!quant.uniform<i8:f32, 0.5:-2>>, tensor<4x8x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 8], scales = dense<[[0.5], [1.0], [1.5], [0.25]]> : tensor<4x1xf16>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, tensor<4x!quant.uniform<i32:f32:0, {0.25, 0.5, 0.75, 0.125}>>) -> tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
  func.return %out : tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>
}
