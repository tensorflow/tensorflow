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

// Verify that when an input and output share the same name ("x"),
// the signature_def maps the input to tensor index 0 and the output to
// the computed output tensor index (1) rather than aliasing to 0.

// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1, 4 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "x",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 4 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "x",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 1 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 0 ],
// CHECK-NEXT:      outputs: [ 1 ],
// CHECK-NEXT:      builtin_options_type: AddOptions,
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
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 3
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [ {
// CHECK-NEXT:    inputs: [ {
// CHECK-NEXT:      name: "x"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    outputs: [ {
// CHECK-NEXT:      name: "x",
// CHECK-NEXT:      tensor_index: 1
// CHECK-NEXT:    } ],
// CHECK-NEXT:    signature_key: "serving_default"
// CHECK-NEXT:  } ]
// CHECK-NEXT:}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "x", outputs = "x"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tfl.add"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    func.return %0 : tensor<1x4xf32>
  }
}
