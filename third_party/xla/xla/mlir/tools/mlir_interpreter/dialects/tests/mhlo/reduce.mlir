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

func.func @reduce() -> tensor<3xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %init = mhlo.constant dense<1> : tensor<i32>
  %reduce = mhlo.reduce(%cst init: %init) across dimensions = [0]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
    reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %0 = mhlo.add %arg0, %arg1 : tensor<i32>
      mhlo.return %0 : tensor<i32>
    }
  return %reduce : tensor<3xi32>
}

// CHECK-LABEL: @reduce
// CHECK-NEXT: Results
// CHECK-NEXT: [6, 8, 10]
