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

// CHECK: error: 'tf.Div' op is neither a custom op nor a flex op
// CHECK: error: failed while converting: 'main'
// CHECK: Some ops are not supported by the native TFLite runtime
// CHECK: tf.Div(tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>) : {name = "div"}

func.func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // tf.div is the result of conversion to a Flex TF op
  %2 = "tf.Div"(%1, %0) {name = "div"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "tfl.exp"(%2) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  func.return %3 : tensor<4xf32>
}
