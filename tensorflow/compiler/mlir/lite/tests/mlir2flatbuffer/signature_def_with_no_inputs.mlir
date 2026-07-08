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

// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [  ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 5 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "StatefulPartitionedCall:1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [  ],
// CHECK-NEXT:    outputs: [ 0 ],
// CHECK-NEXT:    operators: [  ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 2
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [ {
// CHECK-NEXT:    inputs: [  ],
// CHECK-NEXT:    outputs: [ {
// CHECK-NEXT:      name: "start_logits"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    signature_key: "serving_default"
// CHECK-NEXT:  } ]
// CHECK-NEXT: }

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func.func @main() -> (tensor<5xf32> {tf_saved_model.index_path = ["start_logits"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "StatefulPartitionedCall:1"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<5xf32>
    func.return %cst : tensor<5xf32>
  }
}
