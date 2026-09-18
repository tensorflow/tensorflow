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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --use-buffer-offset %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --use-buffer-offset -disable-buffer-deduping %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// Tests that a resource constant referenced by multiple ExportBuffers
// (direct constant usage and streaming cast) does not get prematurely
// released on the first buffer, triggering an error on subsequent buffers.

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<2x2xbf16>) -> (tensor<2x2xbf16>, tensor<2x2xf32>) {
  // CHECK: value = dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xbf16>
  %0 = arith.constant dense_resource<res_bf16> : tensor<2x2xbf16>
  // CHECK: value = dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  func.return %0, %1 : tensor<2x2xbf16>, tensor<2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_bf16: "0x40000000803F004040408040"
    }
  }
#-}
