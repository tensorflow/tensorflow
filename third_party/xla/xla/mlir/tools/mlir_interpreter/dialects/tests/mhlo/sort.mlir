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

func.func @sort() -> (tensor<2x5xf32>, tensor<2x5xi32>) {
  %input0 = arith.constant dense<[
    [4.0, 2.0, 1.0, 5.0, 3.0],
    [6.0, 9.0, 8.0, 7.0, 10.0]
  ]> : tensor<2x5xf32>

  %input1 = arith.constant dense<[
    [1, 2, 3, 4,  5],
    [6, 7, 8, 9, 10]
  ]> : tensor<2x5xi32>
  %0, %1 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5xf32>, tensor<2x5xi32>) -> (tensor<2x5xf32>, tensor<2x5xi32>)

  return %0, %1 : tensor<2x5xf32>, tensor<2x5xi32>
}

// CHECK-LABEL: @sort
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]
// CHECK-NEXT{LITERAL}: [[3, 2, 5, 1, 4], [6, 9, 8, 7, 10]]
