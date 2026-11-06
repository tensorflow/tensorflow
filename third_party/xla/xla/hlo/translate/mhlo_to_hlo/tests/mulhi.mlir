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

// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>, %arg2: tensor<4xui32>, %arg3: tensor<4xui32>) -> tuple<tensor<4xi32>, tensor<4xui32>> {
  // CHECK: s32[4] mulhi(
  // CHECK: u32[4] mulhi(
  %0 = "mhlo.mulhi"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "mhlo.mulhi"(%arg2, %arg3) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
  %2 = "mhlo.tuple"(%0, %1) : (tensor<4xi32>, tensor<4xui32>) -> tuple<tensor<4xi32>, tensor<4xui32>>
  func.return %2 : tuple<tensor<4xi32>, tensor<4xui32>>
}
