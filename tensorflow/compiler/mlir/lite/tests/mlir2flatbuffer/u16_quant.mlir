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

func.func @main(%arg0: tensor<*x!quant.uniform<u16:f32, 2.0:37>>) -> tensor<*x!quant.uniform<u16:f32, 2.0:37>> {
// CHECK:     {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      type: UINT16,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 2.0 ],
// CHECK-NEXT:        zero_point: [ 37 ]
// CHECK:           }
// CHECK-NEXT:    } ],
  return %arg0 : tensor<*x!quant.uniform<u16:f32, 2.0:37>>
}
