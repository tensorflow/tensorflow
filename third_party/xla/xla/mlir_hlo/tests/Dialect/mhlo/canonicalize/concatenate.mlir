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
// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: func @single_operand
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @single_operand(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.concatenate"(%arg) <{dimension = 0 : i64}> : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @operand_with_dynamic_shape
func.func @operand_with_dynamic_shape(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: mhlo.concatenate
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{dimension = 0 : i64}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK-NEXT: return
  func.return %0 : tensor<?xf32>
}
