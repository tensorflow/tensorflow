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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --use-buffer-offset %s -o - | flatbuffer_to_string - | FileCheck %s

// Custom ops in different subgraphs that share the same input and output tensor
// indices must each get their own large_custom_options_offset and size. The
// distinct large_custom_options_size values (4 vs 8) verify that the two ops
// are matched to their respective entries rather than the same entry.

module {
func.func @sub1(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}} {
  %0 = "tfl.custom"(%arg0) {custom_code = "MyCustomOp", custom_option = #tfl<const_bytes : "0x01020304">} : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

func.func @sub2(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}} {
  %0 = "tfl.custom"(%arg0) {custom_code = "MyCustomOp", custom_option = #tfl<const_bytes : "0x05060708090a0b0c">} : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
}

// CHECK:      operators: [ {
// CHECK-NEXT:   inputs: [ 0 ],
// CHECK-NEXT:   outputs: [ 1 ],
// CHECK-NEXT:   large_custom_options_offset: {{[1-9][0-9]+}},
// CHECK-NEXT:   large_custom_options_size: 4
// CHECK-NEXT: } ],
// CHECK-NEXT: name: "sub1"
// CHECK:      operators: [ {
// CHECK-NEXT:   inputs: [ 0 ],
// CHECK-NEXT:   outputs: [ 1 ],
// CHECK-NEXT:   large_custom_options_offset: {{[1-9][0-9]+}},
// CHECK-NEXT:   large_custom_options_size: 8
// CHECK-NEXT: } ],
// CHECK-NEXT: name: "sub2"
