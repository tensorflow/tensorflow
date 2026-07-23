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
// RUN: tfg-transforms-opt %s --pass-pipeline='builtin.module(tfg-functional-to-region,tfg-region-to-functional,tfg-functional-to-region,tfg-region-to-functional,tfg-functional-to-region,tfg-region-to-functional)' \
// RUN: | FileCheck %s

// Check that functions are renamed at most once when passed through the region
// conversion multiple times, where the first pass re-orders the arguments.

// CHECK: tfg.func @then
tfg.func @then(%arg0: tensor<i32>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i32>) {
  return(%arg1, %arg0) : tensor<i64>, tensor<i32>
}

// CHECK: tfg.func @else
tfg.func @else(%arg0: tensor<i32>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i32>) {
  return(%arg1, %arg0) : tensor<i64>, tensor<i32>
}

// CHECK-LABEL: tfg.func @test
// CHECK-NEXT: %[[ARG1:.*]]: tensor<i32>
// CHECK-NEXT: %[[ARG2:.*]]: tensor<i64>
tfg.func @test(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i64>) -> (tensor<i32>, tensor<i64>) {
  // CHECK: If(%{{.*}}, %[[ARG2]], %[[ARG1]])
  // CHECK-SAME: else_branch = #tf_type.func<@else_1, {}>
  // CHECK-SAME: then_branch = #tf_type.func<@then_0, {}>
  // CHECK-SAME: (tensor<i1>, tensor<i64>, tensor<i32>) ->
  %If:2, %ctl = If(%arg0, %arg1, %arg2) {
    Tcond = i1, Tin = [i32, i64], Tout = [i64, i32],
    output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<i32>, tensor<i64>) -> (tensor<i64>, tensor<i32>)
  return(%If#1, %If#0) : tensor<i32>, tensor<i64>
}

// CHECK: tfg.func @then_0
// CHECK: tfg.func @else_1
// CHECK-NOT: tfg.func
