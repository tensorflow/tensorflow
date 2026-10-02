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

module attributes {tfl.metadata = {"keep_stablehlo_constant" = "true"}} {
 func.func @main () -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<0.000000e+00> : tensor<f32>>}> : () -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
 }
}

//CHECK: func.func @main() -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {outputs = "vhlo.constant_v1"}} {
//CHECK-NEXT:  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<0.000000e+00> : tensor<1x1x1x96xf32>>}> : () -> tensor<1x1x1x96xf32>
//CHECK-NEXT:  return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: }