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

// Exports constants with the TFLite element types `!tfl.string` and
// `!tfl.quint8`.

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main() -> (tensor<2x!tfl.string>, tensor<2x!tfl.quint8>, tensor<2x!tfl.string>) {
  %0 = "tfl.pseudo_const"() {value = dense<["a", "bc"]> : tensor<2x!tfl.string>} : () -> tensor<2x!tfl.string>
  %1 = "tfl.pseudo_const"() {value = dense_resource<quint8_blob> : tensor<2x!tfl.quint8>} : () -> tensor<2x!tfl.quint8>
  %2 = "tfl.pseudo_const"() {value = dense<"a"> : tensor<2x!tfl.string>} : () -> tensor<2x!tfl.string>
  func.return %0, %1, %2 : tensor<2x!tfl.string>, tensor<2x!tfl.quint8>, tensor<2x!tfl.string>
}

{-#
  dialect_resources: {
    builtin: {
      quint8_blob: "0x040000000102"
    }
  }
#-}

// CHECK:      shape: [ 2 ],
// CHECK-NEXT: type: STRING,
// CHECK-NEXT: buffer: [[STR_BUF:[0-9]+]],
// CHECK:      shape: [ 2 ],
// CHECK-NEXT: type: UINT8,
// CHECK-NEXT: buffer: [[QUINT8_BUF:[0-9]+]],
// CHECK:      buffers: [
// String buffer: count 2, offsets 16, 17, 19 (little endian), then "abc".
// CHECK:      data: [ 2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 97, 98, 99 ]
// CHECK:      data: [ 1, 2 ]
// Splat string constant: both elements are "a".
// CHECK:      data: [ 2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 97, 97 ]
