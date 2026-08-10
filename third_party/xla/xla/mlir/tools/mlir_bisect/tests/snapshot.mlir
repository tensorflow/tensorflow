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
// RUN: not mlir-bisect %s --hlo-snapshot=%s.pb \
// RUN: --pass-pipeline="builtin.module(test-break-linalg-transpose)" \
// RUN: | FileCheck %s

func.func @main(%a: tensor<3x1xi32>, %b: tensor<3x1xi32>) -> tensor<3x1xi32> {
  return %a : tensor<3x1xi32>
}

// CHECK: initial module
// CHECK: func @main() -> tensor<3x1xi32> {
// CHECK{LITERAL}: arith.constant dense<[[2], [-4], [5]]> : tensor<3x1xi32>
// CHECK{LITERAL}: arith.constant dense<[[0], [7], [-5]]> : tensor<3x1xi32>
