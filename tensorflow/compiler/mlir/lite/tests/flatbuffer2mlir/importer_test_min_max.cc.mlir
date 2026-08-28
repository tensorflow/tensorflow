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
// This only test the exporter and importer are working without min/max quantization parameters.

func.func @main(tensor<40x37xf32>, tensor<40x37xf32>) -> tensor<40x40xf32> {
^bb0(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>):
  %cst = arith.constant dense<1.0> : tensor<40xf32>
  %0:2 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  func.return %0 : tensor<40x40xf32>

// CHECK-LABEL: func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32>
// CHECK:      %[[CONST:[0-9]+]] = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<40xf32>}>
// CHECK-NEXT: %[[FULL:[0-9]+]]:2 = "tfl.fully_connected"(%arg0, %arg1, %[[CONST]]) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK-NEXT: return %[[FULL]]#0
}
