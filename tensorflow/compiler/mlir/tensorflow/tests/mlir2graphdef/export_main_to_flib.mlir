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
// RUN: tf-mlir-translate -mlir-to-graphdef -tf-export-entry-func-to-flib  %s -o - 2>&1 | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 458 : i32}} {
  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}} {
    tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "TPU:0", name = "const", dtype = "tfdtype$DT_INT32", value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
      tf_executor.fetch
    }
    func.return
  }
}

// CHECK-NOT: node

// CHECK: library
// CHECK-NEXT: function
// CHECK-NEXT: signature
// CHECK-NEXT: name: "main"
// CHECK: node_def
// CHECK: op: "Const"
