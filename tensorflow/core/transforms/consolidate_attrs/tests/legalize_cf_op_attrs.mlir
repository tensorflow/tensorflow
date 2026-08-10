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
// RUN: tfg-transforms-opt --split-input-file --tfg-prepare-attrs-export %s | FileCheck %s

// CHECK-LABEL: tfg.func @test_if_attrs
tfg.func @test_if_attrs(%cond: tensor<i1>, %arg: tensor<i32>) -> (tensor<4xi32>) {
  // CHECK: If
  // CHECK-SAME: Tcond = i1, Tin = [i32], Tout = [i32]
  %If, %ctl = If(%cond, %arg) [%cond.ctl] {
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<i32>) -> (tensor<4xi32>)
  return(%If) : tensor<4xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_case_attrs
tfg.func @test_case_attrs(%branch: tensor<i32>, %arg: tensor<i32>) -> (tensor<2xi32>) {
  // CHECK: Case
  // CHECK-SAME: Tin = [i32], Tout = [i32]
  %Case, %ctl = Case(%branch, %arg) [%branch.ctl] {branches = []} : (tensor<i32>, tensor<i32>) -> (tensor<2xi32>)
  return(%Case) : tensor<2xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_while_attrs
tfg.func @test_while_attrs(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: While
  // CHECK-SAME: T = [i32]
  %While, %ctl = While(%arg) {
    body = #tf_type.func<@body, {}>, cond = #tf_type.func<@cond, {}>,
    parallel_iterations = 10 : i64
  } : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_for_attrs
tfg.func @test_for_attrs(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: For
  // CHECK: T = [i32]
  %For, %ctl = For(%arg, %arg, %arg, %arg) [%arg.ctl] {body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%For) : tensor<i32>
}
