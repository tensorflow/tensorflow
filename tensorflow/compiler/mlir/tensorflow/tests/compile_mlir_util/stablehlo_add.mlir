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


// TODO(b/259459405): Remove this test along with the upstream refactoring to
// avoid non TF inputs.
// This is not a supported mode.
module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> (f32[]) {
// CHECK-NEXT:    %[[ARG0]] = f32[] parameter(0)
// CHECK-NEXT:    %[[ARG1]] = f32[] parameter(1)
// CHECK-NEXT:    [[ADD:%.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])
// CHECK-NEXT:    ROOT %tuple.{{[0-9]+}} = (f32[]) tuple(f32[] [[ADD]])
// CHECK-NEXT:  }
