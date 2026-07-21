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

// CHECK-LABEL: tfg.func @test_drop_dtype(
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>)
tfg.func @test_drop_dtype(%arg0: tensor<i32> {tfg.dtype = i32}) -> (tensor<i32>) {
  return(%arg0) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_drop_is_ref(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.int32ref>)
tfg.func @test_drop_is_ref(%arg0: tensor<*x!tf_type.int32ref> {tfg.is_ref}) -> (tensor<*xi32>) {
  %DeRef, %ctl = DeRef(%arg0) : (tensor<*x!tf_type.int32ref>) -> (tensor<*xi32>)
  return(%DeRef) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_skip_ctl
// CHECK-SAME: tfg.name = "a"
// CHECK-NEXT: tfg.name = "b"
// CHECK-NEXT: tfg.name = "c"
tfg.func @test_skip_ctl(%a: tensor<*xi32> {tfg.name = "a"},
                        %b: tensor<*xi32> {tfg.name = "b"})
    -> (tensor<*xi32> {tfg.name = "c"}) {
  return(%a) : tensor<*xi32>
}
