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
// RUN: tfg-transforms-opt --tfg-functional-to-region --tfg-region-to-functional %s | FileCheck %s

// Check that roundtripping through conversion where the function is re-used
// does not change the op names.

// CHECK-LABEL: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Case
  // CHECK-SAME: @case
  %Case, %ctl = Case(%arg0, %arg1) name("case") {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// CHECK: tfg.func @case(
tfg.func @case(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: A(
  // CHECK-SAME: name("foo")
  // CHECK-SAME: {_some_attr = 5 : i32}
  %A, %ctl = A(%arg0) name("foo") {_some_attr = 5 : i32} : (tensor<i32>) -> (tensor<i32>)
  return(%A) : tensor<i32>
}

// CHECK-NOT: tfg.func
