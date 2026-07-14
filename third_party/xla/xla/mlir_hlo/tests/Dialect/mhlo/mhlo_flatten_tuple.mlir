// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: mlir-hlo-opt -split-input-file -mhlo-flatten-tuple %s | FileCheck %s

// CHECK-LABEL: @custom_call
// CHECK-SAME: %[[X:.*]]: tensor<6x3xf32>
// CHECK: %[[CALL:.+]]:2 = mhlo.custom_call @f(%[[X]]) {api_version = 2 : i32} : (tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) 
// CHECK: return %[[CALL]]#0, %[[CALL]]#1 : tensor<6xf32>, tensor<3xf32> 
func.func @custom_call(%x: tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) {
  %0 = "mhlo.custom_call"(%x) {api_version = 2 : i32, call_target_name = "f"}
    : (tensor<6x3xf32>) -> tuple<tensor<6xf32>, tensor<3xf32>>
  %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<6xf32>
  %2 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %1, %2 : tensor<6xf32>, tensor<3xf32>
}

// -----

// CHECK-LABEL: @custom_call_tupled_operand
// CHECK-NOT: mhlo.tuple
func.func @custom_call_tupled_operand(%arg: tuple<tensor<ui32>, tensor<i32>>)
  -> (tensor<i32>, tensor<ui32>) {
  %0 = mhlo.constant dense<1> : tensor<ui32>
  %1 = mhlo.constant dense<10> : tensor<i32>
  %2 = mhlo.tuple %0, %1, %arg : tuple<tensor<ui32>, tensor<i32>,
                                       tuple<tensor<ui32>, tensor<i32>>>
  %3 = mhlo.custom_call @ScalarProgramDummyConstant(%2)
    : (tuple<tensor<ui32>, tensor<i32>, tuple<tensor<ui32>, tensor<i32>>>)
    -> tensor<ui32>
  return %1, %3 : tensor<i32>, tensor<ui32>
}
