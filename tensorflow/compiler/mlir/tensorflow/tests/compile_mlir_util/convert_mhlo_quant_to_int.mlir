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
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=: -tf-xla-emit-return-tuple | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.uniform_quantize"(%arg0) : (tensor<f32>) -> tensor<!quant.uniform<ui8:f32, 34.0:16>>
    %1 = "stablehlo.uniform_dequantize"(%0) : (tensor<!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<f32>
    func.return %1 : tensor<f32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG0:.*]]: f32[]) -> (f32[]) {
// CHECK:         %[[DIV:.*]] = f32[] divide(f32[] %[[ARG0]],
// CHECK:         %[[ADD:.*]] = f32[] add(f32[] %[[DIV]], f32[]
// CHECK:         %[[CLAMP:.*]] = f32[] clamp(f32[] %[[MIN:.*]], f32[] %[[ADD]],
// CHECK:         %[[ROUND:.*]] = f32[] round-nearest-even(f32[] %[[CLAMP]])
// CHECK:         %[[CONVERT_0:.*]] = u8[] convert(f32[] %[[ROUND]])
// CHECK:         %[[CONVERT_1:.*]] = s32[] convert(u8[] %[[CONVERT_0]])
// CHECK:         %[[SUB:.*]] = s32[] subtract(s32[] %[[CONVERT_1]],
// CHECK:         %[[CONVERT_2:.*]] = f32[] convert(s32[] %[[SUB]])
// CHECK:         %[[MUL:.*]] = f32[] multiply(f32[] %[[CONVERT_2]],
// CHECK:         ROOT %[[TUPLE:.*]] = (f32[]) tuple(f32[] %[[MUL]])
