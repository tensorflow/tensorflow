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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --use-buffer-offset %s -o - | flatbuffer_to_string - | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --use-buffer-offset -disable-buffer-deduping %s -o - | flatbuffer_to_string - | FileCheck %s --check-prefix=NO_DEDUPE

// Constants held in distinct attributes but with identical data (e.g. the same
// weight materialized separately per signature) must share one buffer index,
// not just one payload offset.

module {
func.func @add(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}} {
  %0 = "tfl.pseudo_const"() {value = dense_resource<res_a> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "tfl.add"(%0, %arg0) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}

func.func @sub(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}} {
  %0 = "tfl.pseudo_const"() {value = dense_resource<res_b> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "tfl.sub"(%0, %arg0) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}
}

{-#
  dialect_resources: {
    builtin: {
      res_a: "0x040000000000803F000000400000404000008040",
      res_b: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// CHECK:      name: "x",
// CHECK:      buffer: [[BUF:[0-9]+]],
// CHECK-NEXT: name: "tfl.pseudo_const",
// CHECK:      name: "add"
// CHECK:      name: "x",
// CHECK:      buffer: [[BUF]],
// CHECK-NEXT: name: "tfl.pseudo_const1",
// CHECK:      name: "sub"

// Without deduping, each constant keeps its own buffer.
// NO_DEDUPE:      name: "x",
// NO_DEDUPE:      buffer: 2,
// NO_DEDUPE-NEXT: name: "tfl.pseudo_const",
// NO_DEDUPE:      name: "x",
// NO_DEDUPE:      buffer: 5,
// NO_DEDUPE-NEXT: name: "tfl.pseudo_const1",
