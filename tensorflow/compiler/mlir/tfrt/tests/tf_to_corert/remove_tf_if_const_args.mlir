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
// RUN: tf-tfrt-opt -tfrt-remove-tf-if-const-args %s | FileCheck %s -dump-input-filter=all

func.func @then(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @else(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%x, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%y, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

// CHECK-LABEL: func private @then_removed_const_args_0
// CHECK-SAME: ([[x:%.*]]: tensor<i32>)
// CHECK: [[const:%.*]] = "tf.Const"
// CHECK-SAME: value = dense<10> : tensor<i32>
// CHECK: [[r:%.*]] = "tf.StatefulPartitionedCall"([[x]], [[const]])
// CHECK-SAME: f = @then}
// CHECK: return [[r]]

// CHECK-LABEL: func private @else_removed_const_args_0
// CHECK-SAME: ([[x:%.*]]: tensor<i32>)
// CHECK: [[const:%.*]] = "tf.Const"
// CHECK-SAME: value = dense<10> : tensor<i32>
// CHECK: [[r:%.*]] = "tf.StatefulPartitionedCall"([[x]], [[const]])
// CHECK-SAME: f = @else}
// CHECK: return [[r]]

// CHECK-LABEL: func @remove_const_args
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>)
func.func @remove_const_args(%x: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<10> : tensor<i32> } : () -> tensor<i32>
  // CHECK: [[res:%.*]] = "tf.If"([[cond]], [[x]])
  // CHECK-SAME: {else_branch = @else_removed_const_args_0, is_stateless = false, then_branch = @then_removed_const_args_0}
  // CHECK-NEXT: return [[res]]
  %1 = "tf.If"(%cond, %x, %0) {else_branch = @else, then_branch = @then, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @multiple_uses
func.func @multiple_uses(%x: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<10> : tensor<i32> } : () -> tensor<i32>
  // CHECK: [[res:%.*]] = "tf.If"
  // CHECK-SAME: {else_branch = @else_removed_const_args_1, is_stateless = false, then_branch = @then_removed_const_args_1}
  %1 = "tf.If"(%cond, %0, %x) {else_branch = @else, then_branch = @then, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}
