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

func.func @main(tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16xf32>):
  // CHECK:      {
  // CHECK-NEXT:   version: 3,
  // CHECK-NEXT:   operator_codes: [ {
  // CHECK-NEXT:     deprecated_builtin_code: 1,
  // CHECK-NEXT:     version: 1,
  // CHECK-NEXT:     builtin_code: AVERAGE_POOL_2D
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   subgraphs: [ {
  // CHECK-NEXT:     tensors: [ {
  // CHECK-NEXT:       shape: [ 1, 6, 6, 16 ],
  // CHECK-NEXT:       buffer: 1,
  // CHECK-NEXT:       name: "arg0",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       },
  // CHECK-NEXT:       has_rank: true
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 1, 1, 1, 16 ],
  // CHECK-NEXT:       buffer: 2,
  // CHECK-NEXT:       name: "avgpool",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       },
  // CHECK-NEXT:       has_rank: true
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     inputs: [ 0 ],
  // CHECK-NEXT:     outputs: [ 1 ],
  // CHECK-NEXT:     operators: [ {
  // CHECK-NEXT:       inputs: [ 0 ],
  // CHECK-NEXT:       outputs: [ 1 ],
  // CHECK-NEXT:       builtin_options_type: Pool2DOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-NEXT:         padding: VALID,
  // CHECK-NEXT:         stride_w: 1,
  // CHECK-NEXT:         stride_h: 3,
  // CHECK-NEXT:         filter_width: 6,
  // CHECK-NEXT:         filter_height: 3
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ]
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   description: "MLIR Converted.",
  // CHECK-NEXT:   buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   metadata: [ {
  // CHECK-NEXT:   name: "min_runtime_version",
  // CHECK-NEXT:   buffer: 3
  // CHECK-NEXT:   } ]
  // CHECK-NEXT:   signature_defs: [ ]
  // CHECK-NEXT: }

  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> loc("avgpool")
  func.return %0 : tensor<1x1x1x16xf32>
}
