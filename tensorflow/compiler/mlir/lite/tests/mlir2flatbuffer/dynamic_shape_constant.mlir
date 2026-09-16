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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string -

func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %cst = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<?xi32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<?xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}


// CHECK:    tensors: [ {
// CHECK-NEXT:      shape: [ 2 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "tfl.pseudo_const",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true

// CHECK:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 1, 0, 0, 0, 2, 0, 0, 0 ]
// CHECK-NEXT:   }, {
