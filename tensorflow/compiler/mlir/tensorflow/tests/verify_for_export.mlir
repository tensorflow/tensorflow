// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s -split-input-file -tf-verify-for-export -verify-diagnostics | FileCheck %s

module {
  func.func @failsNoIslands() {
    // expected-error @+1 {{functions must be of a single Graph with single op Islands: first op in function is not a tf_executor.graph}}
    func.return
  }
}

// -----

module {
  // CHECK-LABEL: func @passesSingleIslandOp
  func.func @passesSingleIslandOp() {
    // CHECK: _class = ["loc:@class"]
    tf_executor.graph {
      %c, %control0 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %a, %control1 = tf_executor.island wraps "tf.A"() {_class = ["loc:@class"]} : () -> (tensor<2xf32>)
      %s:2, %control2 = tf_executor.island wraps "tf.Split"(%c, %a) {num_split = 2 : i32} : (tensor<i32>, tensor<2xf32>) -> (tensor<1xf32>, tensor<1xf32>)
      tf_executor.fetch
    }
    func.return
  }
}