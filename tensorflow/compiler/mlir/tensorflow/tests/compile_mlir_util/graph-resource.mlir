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
// RUN: tf-mlir-translate -mlir-tf-graph-to-hlo-text %s -tf-input-shapes=2:2 -tf-input-data-types=DT_FLOAT,DT_FLOAT -tf-xla-input-types=parameter,resource -tf-xla-emit-return-tuple | FileCheck %s

module attributes {tf.versions = {producer = 511 : i32}} {
  func.func @main(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf_type.resource>) {
    tf_executor.graph {
      %control = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg0) : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> ()
      tf_executor.fetch %control : !tf_executor.control
    }
    func.return
  }
}

// Tests a conversion from Graph (tf_executor dialect MLIR) to MLIR with
// resource arguments.

// CHECK-LABEL: HloModule main, input_output_alias={ {0}: (1, {}, may-alias) }
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG0:.*]]: f32[2], [[ARG1:.*]]: f32[2]) -> (f32[2]) {
// CHECK-NEXT:    %[[ARG1]] = f32[2]{0} parameter(1)
// CHECK-NEXT:    %[[ARG0]] = f32[2]{0} parameter(0)
// CHECK-NEXT:    ROOT %tuple.{{[0-9]+}} = (f32[2]{0}) tuple(f32[2]{0} %[[ARG0]])
// CHECK-NEXT:  }

// CHECK:       // InputMapping {0, 1}
// CHECK-NEXT:  // XlaInputShape f32[2]
// CHECK-NEXT:  // XlaInputShape f32[2]
// CHECK-NEXT:  // XlaOutputShape (f32[2])
// CHECK-NEXT:  // ResourceUpdate input_index=1 type=float shape=(2) modified
