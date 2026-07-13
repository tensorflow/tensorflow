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
// RUN: hlo-translate -split-input-file -mlir-to-hlo %s | FileCheck %s --check-prefix CHECK
// RUN: hlo-translate -split-input-file -mlir-to-hlo -print-large-constants %s | FileCheck %s --check-prefix CHECK-PRINT-LARGE

func.func @main(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  // CHECK-PRINT-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  %0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  func.return %0 : tensor<10xi32>
}

// -----

func.func @main(%arg0: tensor<11xi32>) -> tensor<11xi32> {
  // CHECK: constant({...})
  // CHECK-PRINT-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  %0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<11xi32>
  func.return %0 : tensor<11xi32>
}
