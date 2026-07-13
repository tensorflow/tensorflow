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
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=10,17:17,19 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=10,17:17,19 | FileCheck -check-prefix=NO_TUPLES %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text-via-builder %s -tf-input-shapes=10,17:17,19 | FileCheck -check-prefix=NO_TUPLES %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<*xf32>, %arg1: tensor<?x19xf32>) -> tensor<?x19xf32> {
    %0 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<?x19xf32>) -> tensor<?x19xf32>
    func.return %0 : tensor<?x19xf32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       (arg_tuple.{{[0-9]+}}: (f32[10,17], f32[17,19])) -> (f32[10,19])

// NO_TUPLES-LABEL: HloModule main
// NO_TUPLES:       ({{.+}}: f32[10,17], {{.+}}: f32[17,19]) -> f32[10,19]
