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

func.func @generate() -> tensor<?xindex> {
  %size = arith.constant 5 : index
  %iota = tensor.generate %size {
    ^bb0(%i : index):
      tensor.yield %i : index
    } : tensor<?xindex>
  return %iota : tensor<?xindex>
}

// CHECK-LABEL: @generate
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 1, 2, 3, 4]

func.func @generate_2d() -> tensor<?x?xindex> {
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %iota = tensor.generate %c5, %c2 {
    ^bb0(%i : index, %j : index):
      %sum = arith.addi %i, %j : index
      tensor.yield %sum : index
    } : tensor<?x?xindex>
  return %iota : tensor<?x?xindex>
}

// CHECK-LABEL: @generate_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
