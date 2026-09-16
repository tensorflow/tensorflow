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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -emit-builtin-tflite-ops=false -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK:  version: 3,
// CHECK:  operator_codes: [ {
// CHECK:    deprecated_builtin_code: 32,
// CHECK:    custom_code: "HashTableV2",
// CHECK:    builtin_code: CUSTOM
// CHECK: } ],
// CHECK: subgraphs: [ {
// CHECK:   tensors: [ {
// CHECK:     shape: [  ],
// CHECK:     type: RESOURCE,
// CHECK:     buffer: 1,
// CHECK:     name: "tf.HashTableV2",
// CHECK:     quantization: {
// CHECK-EMPTY
// CHECK:     }
// CHECK:   } ],
// CHECK:   inputs: [  ],
// CHECK:   outputs: [ 0 ],
// CHECK:   operators: [ {
// CHECK:     inputs: [  ],
// CHECK:     outputs: [ 0 ],
// CHECK:     custom_options:
// CHECK:   name: "main"
// CHECK: } ],
// CHECK: description: "MLIR Converted.",
// CHECK: buffers: [ {
// CHECK-EMPTY
// CHECK: }, {
// CHECK-EMPTY
// CHECK: } ]
// CHECK: }

func.func @main() -> tensor<*x!tf_type.resource> {
  %0 = "tf.HashTableV2"() {container = "" , shared_name= "table", use_node_name_sharing = false, key_dtype = i32, value_dtype = i32 } : () -> tensor<*x!tf_type.resource>
  func.return %0 : tensor<*x!tf_type.resource>
}

