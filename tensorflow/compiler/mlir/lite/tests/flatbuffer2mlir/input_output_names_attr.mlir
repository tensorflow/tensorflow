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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// Tests input and output names from FlatBuffer are added to `tf.entry_function` attribute.

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi8>)
// CHECK: attributes {tf.entry_function = {inputs = "input0,input1", outputs = "output0,output1"}}
attributes {tf.entry_function = {inputs = "input0,input1", outputs = "output0,output1"}} {
  %0 = "tfl.neg"(%arg0) : (tensor<4xi8>) -> tensor<4xi8> loc("neg")
  %1 = "tfl.neg"(%arg1) : (tensor<4xi32>) -> tensor<4xi32> loc("neg")
  func.return %1, %0 : tensor<4xi32>, tensor<4xi8>
}
