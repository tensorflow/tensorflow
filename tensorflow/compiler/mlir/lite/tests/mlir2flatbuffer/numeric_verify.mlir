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

// CHECK:      {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 32,
// CHECK-NEXT:    custom_code: "NumericVerify",
// CHECK-NEXT:    builtin_code: CUSTOM
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.1 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "NumericVerify/arg1:1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 0 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 1, 0 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      custom_options:
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:  name: "min_runtime_version",
// CHECK-NEXT:  buffer: 4
// CHECK-NEXT:  } ]
// CHECK-NEXT:  signature_defs: [ ]
// CHECK-NEXT:}

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4x!quant.uniform<u8:f32, 0.1>>) -> tensor<4xf32> {
  "tfl.NumericVerify"(%arg1, %arg0) {tolerance = 0.1 : f32} : (tensor<4x!quant.uniform<u8:f32, 0.1>>, tensor<4xf32>) -> (tensor<4xf32>)
  func.return %arg0 : tensor<4xf32>
}
