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

func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0,input1", outputs = "Add"}} {
  %graph = tf_executor.graph {
    %2:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
    tf_executor.fetch %2 : tensor<10xi32>
  }
  func.return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "input0"
// CHECK-NEXT:   op: "_Arg"
// CHECK:      node {
// CHECK-NEXT:   name: "input1"
// CHECK-NEXT:   op: "_Arg"
// CHECK:      node {
// CHECK-NEXT:   name: "Add{{_.*_1}}"
// CHECK-NEXT:   op: "Add"
// CHECK-NEXT:   input: "input0"
// CHECK-NEXT:   input: "input1"
// CHECK:      node {
// CHECK-NEXT:   name: "Add"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "Add{{_.*_1}}"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: library {
// CHECK-NEXT: }
