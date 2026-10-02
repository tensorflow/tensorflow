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

func.func @static() -> tensor<2x3xi32> {
  %ret = tensor.empty() : tensor<2x3xi32>
  return %ret : tensor<2x3xi32>
}

// CHECK-LABEL: @static
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0], [0, 0, 0]]

func.func @dynamic() -> tensor<2x?x3x?xi32> {
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %ret = tensor.empty(%c5, %c7) : tensor<2x?x3x?xi32>
  return %ret : tensor<2x?x3x?xi32>
}

// CHECK-LABEL: @dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<2x5x3x7xi32>
