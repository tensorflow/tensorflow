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

// Exports a dynamic dimension of a `!tfl.variant` subtype as -1.

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s | flatbuffer_to_string - | FileCheck %s

func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tfl.variant<tensor<?xi32>>> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x02">} : (tensor<i32>, tensor<i32>) -> tensor<!tfl.variant<tensor<?xi32>>>
  func.return %0 : tensor<!tfl.variant<tensor<?xi32>>>
}

// CHECK:         type: VARIANT,
// CHECK:         name: "tfl.custom",
// CHECK:         variant_tensors: [ {
// CHECK-NEXT:      shape: [ -1 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      has_rank: true
