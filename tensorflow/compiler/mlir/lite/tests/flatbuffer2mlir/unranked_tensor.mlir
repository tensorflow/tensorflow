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
// RUN: litert-opt --tfl-legalize-tf-while %s -o - | flatbuffer_translate -mlir-to-tflite-flatbuffer - -o -  | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK-LABEL: main
func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:     %{{.*}} = tfl.add %{{.*}}, %{{.*}} {fused_activation_function = "NONE"} : tensor<*xf32>
  // CHECK:     return %{{.*}} : tensor<*xf32>

  %0 = tfl.add(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}