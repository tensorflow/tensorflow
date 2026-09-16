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
// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Tests for TensorFlow TFRT ops with custom verifiers.

//===--------------------------------------------------------------------===//
//  Test TF operations (tf.*)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func @testPwStreamResults
func.func @testPwStreamResults(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  "tf.PwStreamResults"(%arg0, %arg1) {names = ["foo", "bar"]} : (tensor<f32>, tensor<f32>) -> ()
  return
}

// -----
// CHECK-LABEL: func @test_ifrt_call
func.func @test_ifrt_call(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  %result = "tf.IfrtCall"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, program_id = 1234 : i64, variable_arg_indices = [0 : i32, 1 : i32], variable_names = ["a", "b"]}> : (tensor<?xf32>, tensor<?xf32>) -> (tensor<1x1xf32>)
  func.return
}

// -----
func.func @test_ifrt_call_fail_unsorted_variable_arg_indices(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error@below {{variable_arg_indices must be sorted in ascending order}}
  %result = "tf.IfrtCall"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, program_id = 1234 : i64, variable_arg_indices = [1 : i32, 0 : i32], variable_names = ["a", "b"]}> : (tensor<?xf32>, tensor<?xf32>) -> (tensor<1x1xf32>)
  func.return
}
