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
^bb0:
  // CHECK:      key: "listvalue"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   list {
  // CHECK-NEXT:     s: " \n"
  // CHECK-NEXT:   }
  // CHECK:      key: "value"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   s: " 0\n\000\000"
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder"() {name = "dummy", dtype = "tfdtype$DT_INT32", value = "\200\n\00\00", listvalue = ["\20\0A"]} : () -> tensor<2xi32>
    tf_executor.fetch
  }
  func.return
}
