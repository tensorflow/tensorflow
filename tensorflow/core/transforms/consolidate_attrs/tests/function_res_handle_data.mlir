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
// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_handle_data
// CHECK: -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
tfg.func @test_handle_data(%arg0: tensor<*xi32>)
    -> (tensor<*x!tf_type.resource> {tfg.handle_data = [tensor<4xf32>, tensor<2xi32>]}) {
  // CHECK: %[[A:.*]], %{{.*}} = A(%{{.*}}) : (tensor<*xi32>) -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*x!tf_type.resource>)
  // CHECK: return(%[[A]]) : tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>
  return(%A) : tensor<*x!tf_type.resource>
}

// CHECK-LABEL: tfg.func @test_handle_data_arg_type(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
// CHECK: -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
tfg.func @test_handle_data_arg_type(%arg0: tensor<*x!tf_type.resource>)
    -> (tensor<*x!tf_type.resource> {tfg.handle_data = [tensor<4xf32>, tensor<2xi32>]}) {
  // CHECK: return(%[[ARG0]]) : tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>
  return(%arg0) : tensor<*x!tf_type.resource>
}
