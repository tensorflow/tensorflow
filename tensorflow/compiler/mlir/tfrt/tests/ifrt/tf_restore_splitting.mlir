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
// RUN: tf-tfrt-opt %s -tf-restore-splitting | FileCheck %s

// CHECK-LABEL: func @single_restore
// CHECK-SAME:    (%[[ARG0:.*]]: {{.*}})
func.func @single_restore(%arg0: tensor<!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = "tf.Const"() {value = dense<["foo", "bar"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %1 = "tf.Const"() {value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %2:2 = "tf.RestoreV2"(%arg0, %0, %1) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>)

  // CHECK: %[[FOO_NAME:.*]] = "tf.Const"() <{value = dense<"foo"> : tensor<1x!tf_type.string>}>
  // CHECK: %[[FOO:.*]] = "tf.RestoreV2"(%[[ARG0]], %[[FOO_NAME]], {{.*}})

  // CHECK: %[[BAR_NAME:.*]] = "tf.Const"() <{value = dense<"bar"> : tensor<1x!tf_type.string>}>
  // CHECK: %[[BAR:.*]] = "tf.RestoreV2"(%[[ARG0]], %[[BAR_NAME]], {{.*}})

  // CHECK: return %[[FOO]], %[[BAR]]
  func.return %2#0, %2#1 : tensor<*xf32>, tensor<*xi32>
}
