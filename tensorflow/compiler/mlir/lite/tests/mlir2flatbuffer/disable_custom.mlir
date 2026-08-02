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
// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s

// CHECK: error: 'tf.MyCustomOp' op is neither a custom op nor a flex op
// CHECK: error: failed while converting: 'main'
// CHECK: Some ops in the model are custom ops, See instructions to implement
// CHECK: tf.MyCustomOp(tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<3xf32>) : {name = "MyCustomOp"}

func.func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2:2 = "tf.MyCustomOp"(%1, %0) {name = "MyCustomOp"} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<3xf32>)
  %3 = "tfl.exp"(%2#0) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  func.return %3 : tensor<4xf32>
}
