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
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=: -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=: | FileCheck -check-prefix=NO_TUPLES %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text-via-builder %s -tf-input-shapes=: | FileCheck -check-prefix=NO_TUPLES %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main() -> (tensor<0xi32>, tensor<0xi32>) {
    %0 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
    %r0, %r1 = "tf.BroadcastGradientArgs"(%0, %0) {T = i32} : (tensor<0xi32>, tensor<0xi32>) -> (tensor<0xi32>, tensor<0xi32>)
    func.return %r0, %r1 : tensor<0xi32>, tensor<0xi32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9+]}} ([[ARG_TUPLE:.*]]: ()) -> (s32[0], s32[0]) {
// CHECK:         %[[ARG_TUPLE]] = () parameter(0)
// CHECK:         [[CONSTANT:%.*]] = s32[0]{0} constant({})
// CHECK:         ROOT %tuple.{{[0-9]+}} = (s32[0]{0}, s32[0]{0}) tuple(s32[0]{0} [[CONSTANT]], s32[0]{0} [[CONSTANT]])
// CHECK:       }

// NO_TUPLES-LABEL: HloModule main
// NO_TUPLES:       ENTRY %main.{{[0-9+]}} () -> (s32[0], s32[0]) {
// NO_TUPLES:         [[CONSTANT:%.*]] = s32[0]{0} constant({})
// NO_TUPLES:         ROOT %tuple.{{[0-9]+}} = (s32[0]{0}, s32[0]{0}) tuple(s32[0]{0} [[CONSTANT]], s32[0]{0} [[CONSTANT]])
// NO_TUPLES:       }
