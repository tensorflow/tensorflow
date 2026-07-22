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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() -> tensor<2x3xi32> {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<5> : tensor<2x3xi32>} : () -> tensor<2x3xi32> loc("Const0")
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<4.2> : tensor<4x5xf32>} : () -> tensor<4x5xf32> loc("Const1")
    %2:3 = tf_executor.island wraps "tf.IdentityN"(%0, %1) : (tensor<2x3xi32>, tensor<4x5xf32>) -> (tensor<2x3xi32>, tensor<4x5xf32>) loc("MyIdentityN")
    tf_executor.fetch %2#0 : tensor<2x3xi32>
  }
  func.return %graph : tensor<2x3xi32>
}

// CHECK:        name: "MyIdentityN"
// CHECK-NEXT:   op: "IdentityN"
// CHECK-NEXT:   input: "Const0"
// CHECK-NEXT:   input: "Const1"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         type: DT_INT32
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
