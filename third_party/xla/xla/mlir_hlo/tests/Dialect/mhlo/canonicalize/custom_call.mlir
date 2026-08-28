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
// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL:@noeffect
func.func @noeffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NOT: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo", has_side_effect = false} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

// CHECK-LABEL:@sideeffect
func.func @sideeffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo", has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

// CHECK-LABEL:@defaulteffect
func.func @defaulteffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

