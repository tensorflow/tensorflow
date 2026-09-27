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

func.func @dim() -> index {
  %it = tensor.empty() : tensor<2x3x4xi32>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %it, %c1 : tensor<2x3x4xi32>
  return %dim : index
}

// CHECK-LABEL: @dim
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 3
