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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o -
// TODO(b/329300758): add file check back after the cl is fixed | FileCheck %s
// Check a few basic properties of the import-export,
// including constants retaining their shape
// and the module including the TFLite version.

func.func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  // CHECK: module attributes
  // CHECK-SAME: tfl.description = "MLIR Converted."
  // CHECK-SAME: tfl.schema_version = 3 : i32

  // CHECK:          %{{.*}} = "tfl.pseudo_const"() <{value = dense<{{\[\[1, 2\], \[3, 4\], \[5, 6\]\]}}> : tensor<3x2xi32>}>
  // CHECK-NEXT:     [[SUB:%.*]] = tfl.sub %{{.*}}, %{{.*}} {fused_activation_function = "RELU6"} : tensor<3x2xi32>
  // CHECK-NEXT:     [[SCALAR:%.*]] = "tfl.pseudo_const"() <{value = dense<10> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT:     [[ADD:%.*]] = tfl.add([[SCALAR]], [[SUB]]) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  // CHECK-NEXT:     return [[ADD]] : tensor<3x2xi32>

  %0 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32> loc("Const")
  %1 = "tfl.sub" (%arg0, %0) {fused_activation_function = "RELU6"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("sub")
  %2 = "arith.constant" () {value = dense<10> : tensor<i32>} : () -> tensor<i32> loc("Const2")
  %3 = "tfl.add" (%2, %1) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("add")
  func.return %3 : tensor<3x2xi32>
}
