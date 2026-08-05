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
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @no_op_cast() -> i32 {
  %cst = arith.constant 42 : i32
  %cast = builtin.unrealized_conversion_cast %cst : i32 to i32
  return %cast : i32
}

// CHECK-LABEL: @no_op_cast
// CHECK-NEXT: Results
// CHECK{LITERAL}: 42

func.func @cast_to_dynamic() -> tensor<?xi32> {
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %cast = builtin.unrealized_conversion_cast %cst : tensor<3xi32> to tensor<?xi32>
  return %cast : tensor<?xi32>
}

// CHECK-LABEL: @cast_to_dynamic
// CHECK-NEXT: Results
// CHECK{LITERAL}: [0, 1, 2]
