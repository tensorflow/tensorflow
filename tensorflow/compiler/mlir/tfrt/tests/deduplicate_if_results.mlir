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
// RUN: tf-tfrt-opt -split-input-file -tfrt-deduplicate-if-result %s | FileCheck %s -dump-input=fail

func.func private @then(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %x, %x : tensor<i32>, tensor<i32>
}

func.func private @else(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %y, %y : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: then/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<i32>

// CHECK-LABEL: else/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<i32>

// CHECK-LABEL: @basic
func.func @basic(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: [[r:%.*]] = "tf.If"
  // CHECK-NEXT: return [[r]], [[r]] : tensor<i32>, tensor<i32>
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @else, then_branch = @then, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

func.func private @unmatched_then(%x: tensor<*xi32>, %y: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  func.return %x, %x : tensor<*xi32>, tensor<*xi32>
}

func.func private @unmatched_else(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %y, %y : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: unmatched_then/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<*xi32>

// CHECK-LABEL: unmatched_else/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<i32>

// CHECK-LABEL: @unmatched_then_else_type
func.func @unmatched_then_else_type(%cond: tensor<i1>, %x: tensor<*xi32>, %y: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  // CHECK-NEXT: [[r:%.*]] = "tf.If"
  // CHECK-NEXT: return [[r]], [[r]] : tensor<*xi32>, tensor<*xi32>
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @unmatched_else, then_branch = @unmatched_then, is_stateless = true} : (tensor<i1>, tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  return %0, %1 : tensor<*xi32>, tensor<*xi32>
}
