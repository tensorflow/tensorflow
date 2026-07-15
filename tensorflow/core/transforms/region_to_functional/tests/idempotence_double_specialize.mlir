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
// RUN: tfg-transforms-opt %s \
// RUN: --tfg-functional-to-region --tfg-region-to-functional \
// RUN: --tfg-functional-to-region --tfg-region-to-functional \
// RUN: | FileCheck %s

// Check that function names remain consistent when passed through region
// conversion multiple times. In this case, the first pass will specialize the
// same branch function twice and create two new functions.

// The function is specialized twice, but it must retain unique names.
// CHECK: tfg.func @case
tfg.func @case(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  return(%arg1) : tensor<i32>
}

// CHECK: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: Case
  // CHECK-SAME: name("foo")
  // CHECK-SAME: @case_tfg_region_specialized_foo_0
  %Case0, %ctl0 = Case(%arg0, %arg1, %arg2) name("foo") {
    Tin = [i32, i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: Case
  // CHECK-SAME: name("bar")
  // CHECK-SAME: @case_tfg_region_specialized_bar_0
  %Case1, %ctl1 = Case(%arg0, %arg1, %arg2) name("bar") {
    Tin = [i32, i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case0, %Case1) : tensor<i32>, tensor<i32>
}

// CHECK: tfg.func @case_tfg_region_specialized_foo_0
// CHECK: tfg.func @case_tfg_region_specialized_bar_0
// CHECK-NOT: tfg.func
