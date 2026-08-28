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
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:  operator_codes: [  ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      type: VARIANT,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:      variant_tensors: [ {
// CHECK-NEXT:        shape: [ 2 ],
// CHECK-NEXT:        type: INT32,
// CHECK-NEXT:        has_rank: true
// CHECK-NEXT:      } ]
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 0 ],
// CHECK-NEXT:    operators: [  ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 2
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [  ]
// CHECK-NEXT: }
func.func @main(%arg0 : tensor<!tf_type.variant<tensor<2xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>> {
  func.return %arg0 : tensor<!tf_type.variant<tensor<2xi32>>>
}
