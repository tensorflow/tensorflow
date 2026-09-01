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

func.func @vecmat() -> tensor<2xi32> {
  %lhs = arith.constant dense<[4, 5]> : tensor<2xi32>
  %rhs = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %init = tensor.empty() : tensor<2xi32>
  %ret = linalg.vecmat ins(%lhs, %rhs: tensor<2xi32>, tensor<2x2xi32>)
                       outs(%init: tensor<2xi32>) -> tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @vecmat
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [19, 28]
