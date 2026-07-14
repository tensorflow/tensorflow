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
attributes {tf.entry_function = {inputs = "foo,bar", outputs = "Add"}} {
  %graph = tf_executor.graph {
    // This node would be renamed to bar1 [note: if imported from TF graphdef this would not be possible]
    %2:2 = tf_executor.island wraps "tf.Identity"(%arg1) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
    // The following node would be renamed to bar2
    %3:2 = tf_executor.island wraps "tf.Identity"(%2) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
    %4:2 = tf_executor.island wraps "tf.Add"(%arg0, %3) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
    tf_executor.fetch %4#0 : tensor<10xi32>
  }
  func.return %graph : tensor<10xi32>
}

// CHECK: name: "foo"
// CHECK-NEXT: op: "_Arg"
// CHECK: name: "bar"
// CHECK-NEXT: op: "_Arg"
// CHECK: name: "[[BAR_ID_0:.*]]"
// CHECK-NEXT: op: "Identity"
// CHECK-NEXT: input: "bar"
// CHECK: name: "[[BAR_ID_1:.*]]"
// CHECK-NEXT: op: "Identity"
// CHECK-NEXT: input: "[[BAR_ID_0]]"
// CHECK: name: "Add"
// CHECK-NEXT: op: "_Retval"
// CHECK-NEXT: input: "Add{{_.*_1}}"
