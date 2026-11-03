// Copyright 2026 The OpenXLA Authors.
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

// RUN: sdy_opt %s -sdy-insert-explicit-reshards -sdy-resolve-permutation-factors -sdy-reshard-to-collectives -sdy-pad-for-divisibility -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func.func @inspect_sharding_with_live_users(
// CHECK-SAME:      %arg0: tensor<4x2xf32>
func.func @inspect_sharding_with_live_users(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>})
    -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: @InspectSharding
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %arg0, %arg0 {{.*}}: tensor<4x2xf32>
  // CHECK: return %[[MUL]] : tensor<4x2xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {
    has_side_effect = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = stablehlo.multiply %0, %0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
  } : tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// Dead-result `@InspectSharding` on a sharded block arg or op result without an
// explicit result sharding must not trigger an `all_gather` or crash.
// CHECK-LABEL: func.func @inspect_sharding_sharded_unused_result(
// CHECK-SAME:      %arg0: tensor<4x2xf32>
func.func @inspect_sharding_sharded_unused_result(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>})
    -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: all_gather
  // CHECK-NOT: @InspectSharding
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {{.*}}: tensor<4x2xf32>
  // CHECK-NOT: @InspectSharding
  // CHECK: return %[[ADD]] : tensor<4x2xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {
    has_side_effect = true
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = stablehlo.add %arg0, %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
  } : tensor<8x4xf32>
  %2 = stablehlo.custom_call @InspectSharding(%1) {
    has_side_effect = true
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// Non-`InspectSharding` custom calls fall back to default global-to-local
// partitioning when `@InspectSharding` is erased.
// CHECK-LABEL: func.func @inspect_sharding_preserves_other_custom_call(
// CHECK-SAME:      %arg0: tensor<4x2xf32>
func.func @inspect_sharding_preserves_other_custom_call(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>})
    -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: @InspectSharding
  // CHECK: %[[CALL:.*]] = stablehlo.custom_call @sdy_testonly(%arg0) {{.*}}: (tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK: return %[[CALL]] : tensor<4x2xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {
    has_side_effect = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = stablehlo.custom_call @sdy_testonly(%0) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}
