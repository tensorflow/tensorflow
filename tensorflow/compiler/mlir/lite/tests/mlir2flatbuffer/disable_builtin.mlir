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
// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer -emit-builtin-tflite-ops=false %s 2>&1 | FileCheck %s

// CHECK: 'tfl.add' op is a TFLite builtin op but builtin emission is not enabled

func.func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  %0 = "arith.constant"() {name = "Const2", value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.add"(%0, %arg0) {fused_activation_function = "NONE", name = "add"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}
