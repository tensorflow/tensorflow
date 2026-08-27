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

module attributes {
  tfl.metadata = {key1 = "value1", key2 = "value2"}
} {
  func.func @main(tensor<3x2xi32>) -> tensor<3x2xi32>
    attributes {tf.entry_function = {inputs = "input", outputs = "SameNameAsOutput"}} {
  ^bb0(%arg0: tensor<3x2xi32>):
    %0 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
    %1 = "tfl.sub" (%arg0, %0) {fused_activation_function = "NONE"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
    func.return %1 : tensor<3x2xi32>
  }
}

// CHECK:      buffers: [ {
// CHECK:      }, {
// CHECK:      }, {
// CHECK:      }, {
// CHECK:      }, {
// CHECK-NEXT:   data: [ 118, 97, 108, 117, 101, 49 ]
// CHECK-NEXT: }, {
// CHECK-NEXT:   data: [ 118, 97, 108, 117, 101, 50 ]
// CHECK-NEXT: }, {
// CHECK-NEXT:   data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT: } ],
// CHECK-NEXT: metadata: [ {
// CHECK-NEXT:   name: "key1",
// CHECK-NEXT:   buffer: 4
// CHECK-NEXT: }, {
// CHECK-NEXT:   name: "key2",
// CHECK-NEXT:   buffer: 5
// CHECK-NEXT: }, {
// CHECK-NEXT:   name: "min_runtime_version",
// CHECK-NEXT:   buffer: 6
// CHECK-NEXT: } ]
// CHECK-NEXT: signature_defs: [ ]
// CHECK-NEXT: }
