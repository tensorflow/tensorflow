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
// RUN: tf-tfrt-opt %s -tf-restore-merging | FileCheck %s

// CHECK-LABEL: func @single_restore_group
// CHECK-SAME:    (%[[ARG0:.*]]: {{.*}})
func.func @single_restore_group(%arg0: tensor<!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = "tf.Const"() {value = dense<"foo"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %1 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %2 = "tf.RestoreV2"(%arg0, %0, %1) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<*xf32>

  %3 = "tf.Const"() {value = dense<"bar"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %4 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %5 = "tf.RestoreV2"(%arg0, %3, %4) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<*xi32>

  // CHECK:      %[[NAMES:.*]] = "tf.Const"() <{value = dense<["foo", "bar"]> : tensor<2x!tf_type.string>}>
  // CHECK-NEXT:      %[[SHAPES:.*]] = "tf.Const"() <{value = dense<""> : tensor<2x!tf_type.string>}>
  // CHECK-NEXT:      %[[TENSORS:.*]]:2 = "tf.RestoreV2"(%[[ARG0]], %[[NAMES]], %[[SHAPES]])
  // CHECK-SAME:   -> (tensor<*xf32>, tensor<*xi32>)

  // CHECK:      return %[[TENSORS]]#0, %[[TENSORS]]#1
  func.return %2, %5 : tensor<*xf32>, tensor<*xi32>
}
