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
// Ensure "quantfork.stats" roundtrip exactly

func.func @main(%arg0: tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32> {
// CHECK-LABEL: @main
// CHECK: %[[RES0:.*]] = "quantfork.stats"(%arg0) <{layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>}> : (tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32>

  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32>
  func.return %0 : tensor<1x512x672x8xf32>
}
