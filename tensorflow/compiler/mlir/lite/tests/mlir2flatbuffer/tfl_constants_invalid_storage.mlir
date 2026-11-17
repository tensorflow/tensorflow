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

// Reports a `!tfl.quint8` constant whose storage doesn't match its type.

// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - 2>&1 | FileCheck %s

func.func @main() -> tensor<2x!tfl.quint8> {
  // CHECK: error: unsupported storage for a !tfl.quint8 constant
  %0 = "tfl.pseudo_const"() {value = dense_resource<quint8_blob> : tensor<2x!tfl.quint8>} : () -> tensor<2x!tfl.quint8>
  func.return %0 : tensor<2x!tfl.quint8>
}

{-#
  dialect_resources: {
    builtin: {
      quint8_blob: "0x04000000010203"
    }
  }
#-}
