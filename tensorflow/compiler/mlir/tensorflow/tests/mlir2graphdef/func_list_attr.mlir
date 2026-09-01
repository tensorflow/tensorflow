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

func.func @main() {
  tf_executor.graph {
// CHECK:      node {
// CHECK-NEXT:   name: "predicate"
// CHECK-NEXT:   op: "Const"
// CHECK-NEXT:   attr {
// CHECK:          key: "dtype"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "value"
// CHECK-NEXT:     value {
// CHECK-NEXT:       tensor {
// CHECK-NEXT:         dtype: DT_INT32
// CHECK-NEXT:         tensor_shape {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("predicate")

// CHECK:      node {
// CHECK-NEXT:   name: "Case"
// CHECK-NEXT:   op: "Case"
// CHECK-NEXT:   input: "predicate"
// CHECK:        attr {
// CHECK:          key: "branches"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "foo"
// CHECK-NEXT:         }
// CHECK-NEXT:         func {
// CHECK-NEXT:           name: "bar"
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      }
    %1:2 = tf_executor.island wraps "tf.Case"(%0#0) {Tin = [], Tout = ["tfdtype$DT_FLOAT"], branches = [@foo, @bar], device = "", output_shapes = [], is_stateless = false} : (tensor<i32>) -> tensor<*xf32> loc("Case")
    tf_executor.fetch
  }
  func.return
}

// CHECK-DAG: name: "foo"
func.func @foo() -> tensor<10xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<1.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32> loc("const_1")
    tf_executor.fetch %1#0 : tensor<10xf32>
  }
  func.return %0 : tensor<10xf32>
}

// CHECK-DAG: name: "bar"
func.func @bar() -> tensor<10xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32> loc("const_2")
    tf_executor.fetch %1#0 : tensor<10xf32>
  }
  func.return %0 : tensor<10xf32>
}
