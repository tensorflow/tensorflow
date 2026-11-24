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
// RUN: sdy_opt %s -sdy-convert-global-to-local -split-input-file | FileCheck %s

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_batch_mode_sharded_contracting
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x3x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}, {}]>},
// CHECK-SAME:  %[[ARG2:.*]]: tensor<2xi32>)
// CHECK-SAME: -> (tensor<2x8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>})
func.func @ragged_dot_batch_mode_sharded_contracting(
    %arg0: tensor<4x8x6xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {"y"}]>},
    %arg1: tensor<4x6x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}, {}]>},
    %arg2: tensor<2xi32>)
    -> (tensor<4x8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>}) {
  // CHECK-NOT: mhlo.ragged_dot
  // CHECK-NOT: sdy.sharding_rule
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGH, HIGHEST] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}], unreduced={"y"}>]>} : (tensor<2x8x3xf32>, tensor<2x3x10xf32>) -> tensor<2x8x10xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[DOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[A0]], %[[A1]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<2x8x10xf32>) -> tensor<2x8x10xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<2x8x10xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{
    precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGHEST>],
    ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      lhs_ragged_dimensions = [0], rhs_group_dimensions = []>}> {
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}], unreduced={"y"}>]>,
      sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, k, l], [m])->([i, j, l]) {i=4, j=8, k=6, l=10, m=2} reduction={k} need_replication={m}>
    } : (tensor<4x8x6xf32>, tensor<4x6x10xf32>, tensor<2xi32>) -> tensor<4x8x10xf32>
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_xy, [{"x"}, {}, {}]> : tensor<4x8x10xf32>
  return %1 : tensor<4x8x10xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_non_contracting_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x2x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG2:.*]]: tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<2x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {"y"}]>})
func.func @ragged_dot_non_contracting_divisible(
    %arg0: tensor<4x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>},
    %arg1: tensor<2x4x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"x"}, {}, {"y"}]>},
    %arg2: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>})
    -> (tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {"y"}]>}) {
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>} : (tensor<2x8x4xf32>, tensor<2x2x4x4xf32>, tensor<2x2xi32>) -> tensor<2x8x4xf32>
  // CHECK-NEXT: return %[[RDOT]] : tensor<2x8x4xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>}
    : (tensor<4x8x4xf32>, tensor<2x4x4x8xf32>, tensor<4x2xi32>) -> tensor<4x8x8xf32>
  return %0 : tensor<4x8x8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @pointwise_and_topk_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>}, tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>})
func.func @pointwise_and_topk_divisible(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>})
    -> (tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>},
        tensor<4x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[ERF:.*]] = mhlo.erf %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<2x4xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ERF]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<2x4xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[VALUES:.*]], %[[INDICES:.*]] = mhlo.topk(%[[AG]], k = 3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>, <@mesh_xy, [{"x"}, {}]>]>} : tensor<2x8xf32> -> (tensor<2x3xf32>, tensor<2x3xi32>)
  // CHECK-NEXT: return %[[VALUES]], %[[INDICES]] : tensor<2x3xf32>, tensor<2x3xi32>
  %0 = mhlo.erf %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<4x8xf32>
  %1 = sdy.all_gather [{}, {"y"}] %0 out_sharding=<@mesh_xy, [{"x"}, {}]> : tensor<4x8xf32>
  %values, %indices = mhlo.topk(%1, k = 3, largest = true) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>, <@mesh_xy, [{"x"}, {}]>]>} : tensor<4x8xf32> -> (tensor<4x3xf32>, tensor<4x3xi32>)
  return %values, %indices : tensor<4x3xf32>, tensor<4x3xi32>
}
