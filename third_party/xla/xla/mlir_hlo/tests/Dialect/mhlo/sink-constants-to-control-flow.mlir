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
// RUN: mlir-hlo-opt %s -mhlo-sink-constants-to-control-flow | FileCheck %s

// Tests sinking constants to a while loop.

// CHECK-LABEL: func @sink_const_to_while
func.func @sink_const_to_while(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NEXT: mhlo.while
  // CHECK-SAME: (%[[ITER_ARG:.*]] = %[[ARG1A:.+]]
  %c0 = mhlo.constant dense<1> : tensor<i64>
  %c1 = mhlo.constant dense<2> : tensor<i64>
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<i64>):
    // CHECK: %[[C0:.+]] = mhlo.constant dense<1> : tensor<i64>
    // CHECK: mhlo.compare LT, %[[C0]], %[[ITER_ARG]]
    %1 = "mhlo.compare"(%c0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    // CHECK-DAG: %[[C1:.+]] = mhlo.constant dense<2> : tensor<i64>
    // CHECK-DAG: %[[ADD0:.+]] = mhlo.add %[[ITER_ARG]], %[[ITER_ARG]]
    %2 = mhlo.add %arg1, %arg1 : tensor<i64>
    // CHECK: %[[ADD1:.+]] = mhlo.add %[[C1]], %[[ADD0]]
    %3 = mhlo.add %c1, %2 : tensor<i64>
    // CHECK: %[[ADD2:.+]] = mhlo.add %[[C1]], %[[ADD1]]
    %4 = mhlo.add %c1, %3 : tensor<i64>
    "mhlo.return"(%4) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// Tests sinking constants to a conditional op.

// CHECK-LABEL: func @sink_const_to_conditional
func.func @sink_const_to_conditional(%arg0: tensor<i64>) -> tensor<i64> {
  %c0 = mhlo.constant dense<1> : tensor<i64>
  %c1 = mhlo.constant dense<2> : tensor<i64>
  %0 = "mhlo.compare"(%arg0, %c0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: mhlo.if
  %2 = "mhlo.if"(%0) ({
    // CHECK: %[[C0:.+]] = mhlo.constant dense<1> : tensor<i64>
    // CHECK: %[[ADD0:.+]] = mhlo.add %[[C0]],
    %3 = mhlo.add %c0, %arg0 : tensor<i64>
    "mhlo.return"(%3) : (tensor<i64>) -> ()
  },  {
    // CHECK: %[[C1:.+]] = mhlo.constant dense<2> : tensor<i64>
    // CHECK: %[[ADD1:.+]] = mhlo.add %[[C1]],
    %4 = mhlo.add %c1, %arg0 : tensor<i64>
    "mhlo.return"(%4) : (tensor<i64>) -> ()
  }) : (tensor<i1>) -> tensor<i64>
  func.return %2 : tensor<i64>
}

func.func @sink_const_to_sort(%arg0: tensor<16xf32>) {
  %c0 = arith.constant dense<1.0> : tensor<f32>
  // CHECK: "mhlo.sort"
  %0 = "mhlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: constant dense<1.000000e+00>
    %1 = "mhlo.divide"(%arg1, %c0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.divide"(%arg2, %c0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "mhlo.compare"(%1, %2) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {is_stable = true} : (tensor<16xf32>) -> tensor<16xf32>
  func.return
}
