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

func.func @static() -> tensor<1x2x3xi32> {
  %t = bufferization.alloc_tensor() : tensor<1x2x3xi32>
  return %t : tensor<1x2x3xi32>
}

// CHECK-LABEL: @static
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[0, 0, 0], [0, 0, 0]]]

func.func @dynamic() -> tensor<?x1xi32> {
  %c4 = arith.constant 4 : index
  %t = bufferization.alloc_tensor(%c4) : tensor<?x1xi32>
  return %t : tensor<?x1xi32>
}

// CHECK-LABEL: @dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0], [0], [0], [0]]

func.func @copy() -> tensor<i32> {
  %c = arith.constant dense<123> : tensor<i32>
  %t = bufferization.alloc_tensor() copy(%c) : tensor<i32>
  return %t : tensor<i32>
}

// CHECK-LABEL: @copy
// CHECK-NEXT: Results
// CHECK-NEXT: 123
