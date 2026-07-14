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
// RUN: tfg-transforms-opt --split-input-file --tfg-region-to-functional %s | FileCheck %s

// Check that conversion back to functional form succeeds even if the return
// types are mismatched (but compatible).

// CHECK-LABEL: tfg.func @test_case(
// CHECK-SAME: %[[ARG0:.*]]: tensor
// CHECK-NEXT: %[[ARG1:.*]]: tensor
// CHECK: -> (tensor<*xf32>)
tfg.func @test_case(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<*xf32>) {
  // CHECK: Case(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: (tensor<i32>, tensor<f32>) -> (tensor<*xf32>)
  %Case, %ctl = CaseRegion %arg0 {
    yield(%arg1) : tensor<f32>
  } : (tensor<i32>) -> (tensor<*xf32>)
  return(%Case) : tensor<*xf32>
}

// CHECK-LABEL: tfg.func
// CHECK: -> (tensor<f32>
// CHECK: return
// CHECK-SAME: tensor<f32>

// -----

// CHECK-LABEL: tfg.func @test_while(
// CHECK-SAME: %[[ARG0:.*]]: tensor
// CHECK: -> (tensor<f32>)
tfg.func @test_while(%arg0: tensor<f32>) -> (tensor<f32>) {
  // CHECK: While(%[[ARG0]])
  // CHECK: (tensor<f32>) -> (tensor<f32>)
  // CHECK: return
  // CHECK-SAME: tensor<f32>
  %While, %ctl = WhileRegion(%arg0) {
  ^bb0(%arg1: tensor<*xf32>, %arg2: !tf_type.control):
    %True, %ctl_0 = True : () -> (tensor<*xi1>)
    condition %True : tensor<*xi1> (%arg1) : tensor<*xf32>
  } do {
  ^bb0(%arg1: tensor<*xf32>, %arg2: !tf_type.control):
    yield(%arg1) : tensor<*xf32>
  } {parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<f32>)
  return(%While) : tensor<f32>
}

// CHECK-LABEL: tfg.func
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xf32>
// CHECK: -> (tensor<*xi1>

// CHECK-LABEL: tfg.func
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xf32>
// CHECK: -> (tensor<*xf32>
