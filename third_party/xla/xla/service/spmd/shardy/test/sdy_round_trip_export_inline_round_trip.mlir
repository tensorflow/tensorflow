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
// RUN: sdy_opt %s -xla-sdy-round-trip-export-pipeline -inline -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Test with a nested func op that gets inlined after first export.

// Make sure this temp attr doesn't exist anymore.
// CHECK-NOT: xla.sdy.sharding

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @main(
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}]>})
func.func @main(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}]>}) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ADD_0]], %[[ADD_0]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %[[MUL]] <@mesh, [{?}, {"b", ?}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[SC]], %[[SC]] : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ADD_1]] : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = func.call @nested_func(%0) : (tensor<8x16xf32>) -> (tensor<8x16xf32>)
  %2 = stablehlo.add %1, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-NOT: func @nested_func
func.func @nested_func(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
