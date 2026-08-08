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
// RUN: tfg-transforms-opt --tfg-cf-sink %s | FileCheck %s

// CHECK: tfg.func @test(%[[ARG0:.*]]: tensor<i32>,
// CHECK:                %[[ARG1:.*]]: tensor<i32>,
// CHECK:                %[[ARG2:.*]]: tensor<i1>)
// CHECK:  {
// CHECK:   %[[IF:.*]], %{{.*}} = IfRegion %[[ARG2]] then  {
// CHECK:     %[[ADDV2:.*]], %{{.*}} = AddV2(%[[ARG0]], %[[ARG1]])
// CHECK:     yield(%[[ADDV2]])
// CHECK:   } else {
// CHECK:     %[[SUB:.*]], %{{.*}} = Sub(%[[ARG0]], %[[ARG1]])
// CHECK:     yield(%[[SUB]])
// CHECK:   }
// CHECK:   return(%[[IF]])
// CHECK: }
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %AddV2, %ctl = AddV2(%arg0, %arg1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %Sub, %ctl_0 = Sub(%arg0, %arg1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %IfRegion, %ctl_2 = IfRegion %cond then {
    yield(%AddV2) : tensor<i32>
  } else {
    yield(%Sub) : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)
  return(%IfRegion) : tensor<i32>
}
