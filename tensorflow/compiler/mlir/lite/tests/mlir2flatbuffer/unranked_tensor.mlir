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

module {
func.func @serving_default(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
// CHECK:     {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "serving_default_x",
// CHECK-NEXT:      quantization: {
// CHECK:           }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "outputs",
// CHECK-NEXT:      quantization: {
// CHECK:           }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 1 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 0 ],
// CHECK-NEXT:      outputs: [ 1 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK:           }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK:       }, {
// CHECK:       }, {
// CHECK:       }, {
// CHECK-NEXT:    data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 3
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [  ]
// CHECK-NEXT:}
  %0 = "tfl.add"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
}
