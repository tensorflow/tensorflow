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
// Confirm we can extract type info from reshape

func.func @main() -> tensor<2x2xf32> {
  // CHECK: %[[cst:.*]] = "tfl.pseudo_const"() <{value = dense<2> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK: %{{.*}} = "tfl.reshape"(%{{.*}}, %[[cst]]) : (tensor<4xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.reshape" (%0, %cst) : (tensor<4xf32>, tensor<2xi32>) -> tensor<2x2xf32> loc("reshape")
  func.return %1 : tensor<2x2xf32>
}
