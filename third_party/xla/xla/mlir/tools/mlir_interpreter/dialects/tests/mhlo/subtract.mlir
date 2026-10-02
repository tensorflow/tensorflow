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
// RUN: mlir-interpreter-runner %s | FileCheck %s

func.func @main() -> tensor<2x3xi64> {
  %lhs = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  %rhs = mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi64>
  %result = mhlo.subtract %lhs, %rhs : tensor<2x3xi64>
  return %result : tensor<2x3xi64>
}

// CHECK{LITERAL}: [[-10, -19, -28], [-37, -46, -55]]