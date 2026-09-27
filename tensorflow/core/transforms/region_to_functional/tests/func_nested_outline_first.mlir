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

// Check that a region is not outlined if a potentially re-usable function has
// nested regions in it.

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: %[[INDEX:.*]], %{{.*}} = Index
  %Index, %ctlIndex = Index : () -> (tensor<i32>)
  // CHECK: %[[ARG:.*]], %{{.*}} = Arg
  %Arg, %ctlArg = Arg : () -> (tensor<i32>)
  // CHECK: Case(%[[INDEX]], %[[INDEX]], %[[ARG]])
  // CHECK-SAME: @bar
  %Case, %ctlCase = CaseRegion %Index {
    %Case_0, %ctlCase_0 = CaseRegion %Index {
      yield(%Arg) : tensor<i32>
    } {region_attrs = [#tfg.region_attrs<{sym_name = "foo"} [] [{}]>]} : (tensor<i32>) -> (tensor<i32>)
    yield(%Case_0) : tensor<i32>
  } {region_attrs = [#tfg.region_attrs<{sym_name = "bar"} [] [{}]>]} : (tensor<i32>) -> (tensor<i32>)
}

// CHECK-LABEL: tfg.func @bar
// CHECK-SAME: %[[ARG0:.*]]: tensor
// CHECK-NEXT: %[[ARG1:.*]]: tensor
tfg.func @bar(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Case(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: @foo
  %Case_0, %ctlCase_0 = CaseRegion %arg0 {
    yield(%arg1) : tensor<i32>
  } {region_attrs = [#tfg.region_attrs<{sym_name = "foo"} [] [{}]>]} : (tensor<i32>) -> (tensor<i32>)
  return(%Case_0) : tensor<i32>
}

// CHECK-NOT: tfg.func @bar_0
// CHECK: tfg.func @foo
// CHECK-NOT: tfg.func @bar_0
