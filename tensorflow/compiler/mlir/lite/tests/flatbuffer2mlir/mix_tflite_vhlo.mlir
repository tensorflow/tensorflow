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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --disable-vhlo-to-stablehlo=true - -o - | FileCheck %s
// test stablehlo roundtrip

module {
func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.logistic_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  %1 = "tfl.exp"(%0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> loc("exp")
  func.return %1 : tensor<1x1x1x96xf32>
}
}

// CHECK: func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "exp"}} {
// CHECK-NEXT:  %0 = "vhlo.logistic_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
// CHECK-NEXT:  %1 = "tfl.exp"(%0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
// CHECK-NEXT:  return %1 : tensor<1x1x1x96xf32>
// CHECK-NEXT: }