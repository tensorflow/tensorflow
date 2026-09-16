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

// CHECK-LABEL: func @fold_access
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @fold_access(%arg : tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return [[ARG]]
  %tuple = "mhlo.tuple"(%arg) : (tensor<i32>) -> tuple<tensor<i32>>
  %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  func.return %element : tensor<i32>
}
