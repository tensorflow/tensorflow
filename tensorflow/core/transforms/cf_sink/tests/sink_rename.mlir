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

tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %Add, %ctl = Add(%arg0, %arg1) name("add") {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %Sub, %ctl_0 = Sub(%arg0, %arg1) name("sub") {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: IfRegion
  %IfRegion, %ctl_2 = IfRegion %cond then {
    // CHECK-NEXT: Add
    // CHECK-SAME: name("add_tfg_cf_sunk_if")
    yield(%Add) : tensor<i32>
  // CHECK: else
  } else {
    // CHECK-NEXT: Sub
    // CHECK-SAME: name("sub_tfg_cf_sunk_if")
    yield(%Sub) : tensor<i32>
  } {_mlir_name = "if"} : (tensor<i1>) -> (tensor<i32>)
  return(%IfRegion) : tensor<i32>
}
