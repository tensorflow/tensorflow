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

func.func @transpose() -> tensor<2x3xi32> {
  %a = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %b = tensor.empty() : tensor<2x3xi32>
  %ret = linalg.transpose ins(%a : tensor<3x2xi32>)
                          outs(%b : tensor<2x3xi32>)
                          permutation = [1, 0]
  return %ret : tensor<2x3xi32>
}

// CHECK-LABEL: @transpose
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 3, 5], [2, 4, 6]]

func.func @transpose_bufferized() -> memref<2x3xi32> {
  %a = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : memref<3x2xi32>
  %b = memref.alloc() : memref<2x3xi32>
  linalg.transpose ins(%a : memref<3x2xi32>)
                   outs(%b : memref<2x3xi32>)
                   permutation = [1, 0]
  return %b : memref<2x3xi32>
}

// CHECK-LABEL: @transpose_bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 3, 5], [2, 4, 6]]
