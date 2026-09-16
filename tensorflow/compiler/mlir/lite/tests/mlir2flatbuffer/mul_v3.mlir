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

func.func @main(tensor<3x!quant.uniform<i8:f32, 1.0>>) -> tensor<3x!quant.uniform<i8:f32, 1.0>> {
^bb0(%arg0: tensor<3x!quant.uniform<i8:f32, 1.0>>):
  // CHECK:      {
  // CHECK-NEXT:  version: 3,
  // CHECK-NEXT:  operator_codes: [ {
  // CHECK-NEXT:    deprecated_builtin_code: 18,
  // CHECK-NEXT:    version: 3,
  // CHECK-NEXT:    builtin_code: MUL
  // CHECK-NEXT:  } ],
  // CHECK-NEXT:  subgraphs: [ {
  // CHECK-NEXT:    tensors: [ {
  // CHECK-NEXT:      shape: [ 3 ],
  // CHECK-NEXT:      type: INT8,
  // CHECK-NEXT:      buffer: 1,
  // CHECK-NEXT:      name: "arg0",
  // CHECK-NEXT:      quantization: {
  // CHECK-NEXT:        scale: [ 1.0 ],
  // CHECK-NEXT:        zero_point: [ 0 ]
  // CHECK-NEXT:       },
  // CHECK-NEXT:       has_rank: true
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      shape: [ 3 ],
  // CHECK-NEXT:      type: INT8,
  // CHECK-NEXT:      buffer: 2,
  // CHECK-NEXT:      name: "tfl.pseudo_qconst",
  // CHECK-NEXT:      quantization: {
  // CHECK-NEXT:        scale: [ 1.0 ],
  // CHECK-NEXT:        zero_point: [ 0 ]
  // CHECK-NEXT:       },
  // CHECK-NEXT:       has_rank: true
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      shape: [ 3 ],
  // CHECK-NEXT:      type: INT8,
  // CHECK-NEXT:      buffer: 3,
  // CHECK-NEXT:      name: "mul",
  // CHECK-NEXT:      quantization: {
  // CHECK-NEXT:        scale: [ 1.0 ],
  // CHECK-NEXT:        zero_point: [ 0 ]
  // CHECK-NEXT:       },
  // CHECK-NEXT:       has_rank: true
  // CHECK-NEXT:    } ],
  // CHECK-NEXT:    inputs: [ 0 ],
  // CHECK-NEXT:    outputs: [ 2 ],
  // CHECK-NEXT:    operators: [ {
  // CHECK-NEXT:      inputs: [ 0, 1 ],
  // CHECK-NEXT:      outputs: [ 2 ],
  // CHECK-NEXT:      builtin_options_type: MulOptions,
  // CHECK-NEXT:      builtin_options: {
  // CHECK-EMPTY:
  // CHECK-NEXT:      }
  // CHECK-NEXT:    } ],
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:  } ],
  // CHECK-NEXT:  description: "MLIR Converted.",
  // CHECK-NEXT:  buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:  }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:  }, {
  // CHECK-NEXT:    data: [ 2, 2, 2 ]
  // CHECK-NEXT:  }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:  }, {
  // CHECK-NEXT:    data: [ 49, 46, 49, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  // CHECK-NEXT:  } ],
  // CHECK-NEXT:  metadata: [ {
  // CHECK-NEXT:  name: "min_runtime_version",
  // CHECK-NEXT:  buffer: 4
  // CHECK-NEXT:  } ]
  // CHECK-NEXT:  signature_defs: [ ]
  // CHECK-NEXT:}

  %0 = "tfl.pseudo_qconst"() { qtype = tensor<3x!quant.uniform<i8:f32, 1.0>>, value = dense<2> : tensor<3xi8>} : () -> tensor<3x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<3x!quant.uniform<i8:f32, 1.0>>, tensor<3x!quant.uniform<i8:f32, 1.0>>) -> tensor<3x!quant.uniform<i8:f32, 1.0>> loc("mul")
  func.return %1 : tensor<3x!quant.uniform<i8:f32, 1.0>>
}
