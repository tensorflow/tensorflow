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

func.func @main() -> tensor<3x2xi32> {
  %0 = "tfl.pseudo_const" () {value = dense<0> : tensor<3x2xi32>, tfl.is_variable} : () -> tensor<3x2xi32> loc("variable")
  func.return %0 : tensor<3x2xi32>
}

// CHECK:      {
// CHECK-NEXT:     version: 3,
// CHECK-NEXT:     operator_codes: [ ],
// CHECK-NEXT:     subgraphs: [ {
// CHECK-NEXT:       tensors: [ {
// CHECK-NEXT:         shape: [ 3, 2 ],
// CHECK-NEXT:         type: INT32,
// CHECK-NEXT:         name: "variable",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         },
// CHECK-NEXT:         is_variable: true
// CHECK-NEXT:         has_rank: true
// CHECK-NEXT:       } ],
// CHECK-NEXT:       inputs: [ ],
// CHECK-NEXT:       outputs: [ 0 ],
// CHECK-NEXT:       operators: [ ],
// CHECK-NEXT:       name: "main"
// CHECK-NEXT:     } ],
// CHECK-NEXT:     description: "MLIR Converted.",
// CHECK-NEXT:     buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ {{.*}} ]
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ {{.*}} ]
// CHECK-NEXT:     } ],
// CHECK-NEXT:     metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 2
// CHECK-NEXT:     } ]
// CHECK-NEXT:     signature_defs: [ ]
// CHECK-NEXT:   }