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
// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion | FileCheck %s

// Check ops registered to IO Op Expander only happen on Device 0.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource>) {
  "tf_device.cluster"() ({
    // CHECK:      tf.NotEqual
    // CHECK:      tf.If
    // CHECK-SAME: else_branch = @tf.[[ELSE:[a-zA-Z0-9_]*]]
    // CHECK-SAME: then_branch = @tf.[[THEN:[a-zA-Z0-9_]*]]
    // CHECK:      func private @tf.[[THEN]]
    // CHECK:      tf.NoOp
    // CHECK:      func private @tf.[[ELSE]]
    // CHECK:      "tf.WriteSummary"
    %3 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %2 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<1> : tensor<i64>} : () -> tensor<i64>
    "tf.WriteSummary"(%arg1, %1, %2, %3, %3) {_global_shape = [], device = ""} : (tensor<*x!tf_type.resource>, tensor<i64>, tensor<i32>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()    tf_device.return
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
  func.return
}
