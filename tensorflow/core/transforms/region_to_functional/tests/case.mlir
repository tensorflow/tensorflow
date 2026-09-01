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
// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Test that `CaseRegion` is correctly converted to functional form, which a
// branch function generated for each region.

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INDEX:.*]], %[[CTL:.*]] = Index
  %Index, %ctl = Index : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_0:.*]] = Data
  %Data, %ctl_0 = Data : () -> (tensor<f32>)
  // CHECK-NEXT: %[[CASE:.*]], %[[CTL_1:.*]] = Case(%[[INDEX]], %[[DATA]])
  // CHECK-SAME: {branches = [#tf_type.func<@[[CASE_FUNC_0:.*]], {}>, #tf_type.func<@[[CASE_FUNC_1:.*]], {}>, #tf_type.func<@[[CASE_FUNC_2:.*]], {}>]
  // CHECK-SAME: } : (tensor<i32>, tensor<{{.*}}>) -> (tensor<{{.*}}>)
  %Case, %ctl_1 = CaseRegion %Index  {
    %A, %ctl_2 = A(%Data) : (tensor<f32>) -> (tensor<i32>)
    yield(%A) : tensor<i32>
  },  {
    %B, %ctl_2 = B(%Data) : (tensor<f32>) -> (tensor<i32>)
    yield(%B) : tensor<i32>
  },  {
    %C, %ctl_2 = C(%Data) : (tensor<f32>) -> (tensor<i32>)
    yield(%C) : tensor<i32>
  } : (tensor<i32>) -> tensor<i32>
}

// CHECK: tfg.func @[[CASE_FUNC_0]](%[[ARG0:.*]]: tensor<{{.*}}>
// CHECK:   %[[A:.*]], %[[CTL:.*]] = A(%[[ARG0]])
// CHECK:   return(%[[A]])

// CHECK: tfg.func @[[CASE_FUNC_1]](%[[ARG0:.*]]: tensor<{{.*}}>
// CHECK:   %[[B:.*]], %[[CTL:.*]] = B(%[[ARG0]])
// CHECK:   return(%[[B]])

// CHECK: tfg.func @[[CASE_FUNC_2]](%[[ARG0:.*]]: tensor<{{.*}}>
// CHECK:   %[[C:.*]], %[[CTL:.*]] = C(%[[ARG0]])
// CHECK:   return(%[[C]])
