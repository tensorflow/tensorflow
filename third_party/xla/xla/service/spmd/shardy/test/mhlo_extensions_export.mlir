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
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' -sdy-resolve-permutation-factors -sdy-reshard-to-collectives -sdy-pad-for-divisibility -sdy-convert-global-to-local -split-input-file | FileCheck %s

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @mulhi_non_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<7x5xi32>, %[[ARG1:.*]]: tensor<7x5xi32>) -> tensor<7x5xi32>
func.func @mulhi_non_divisible(%arg0: tensor<7x5xi32>, %arg1: tensor<7x5xi32>) -> tensor<7x5xi32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<7x5xi32>, tensor<i32>) -> tensor<8x6xi32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 3, 0, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[OFF1]], sizes = [4, 3] : (tensor<8x6xi32>, tensor<i32>, tensor<i32>) -> tensor<4x3xi32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<7x5xi32>, tensor<i32>) -> tensor<8x6xi32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 3, 0, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF2]], %[[OFF3]], sizes = [4, 3] : (tensor<8x6xi32>, tensor<i32>, tensor<i32>) -> tensor<4x3xi32>
  // CHECK-NEXT: %[[MULHI:.*]] = mhlo.mulhi %[[SLICE0]], %[[SLICE1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<4x3xi32>
  // CHECK-NEXT: %[[AG_IN:.*]] = stablehlo.reshape %[[MULHI]] : (tensor<4x3xi32>) -> tensor<1x4x3xi32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AG_IN]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<1x4x3xi32>) -> tensor<4x4x3xi32>
  // CHECK-NEXT: %[[AG_R0:.*]] = stablehlo.reshape %[[AG]] : (tensor<4x4x3xi32>) -> tensor<2x2x4x3xi32>
  // CHECK-NEXT: %[[AG_T:.*]] = stablehlo.transpose %[[AG_R0]], dims = [0, 2, 1, 3] : (tensor<2x2x4x3xi32>) -> tensor<2x4x2x3xi32>
  // CHECK-NEXT: %[[AG_R1:.*]] = stablehlo.reshape %[[AG_T]] : (tensor<2x4x2x3xi32>) -> tensor<8x6xi32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG_R1]] [0:7, 0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : (tensor<8x6xi32>) -> tensor<7x5xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<7x5xi32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {"y"}]> : tensor<7x5xi32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {"y"}]> : tensor<7x5xi32>
  %2 = mhlo.mulhi %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<7x5xi32>
  %3 = sdy.reshard %2 <@mesh_xy, [{}, {}]> : tensor<7x5xi32>
  return %3 : tensor<7x5xi32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @pointwise_mhlo_ops_non_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x9xf32>, %[[ARG1:.*]]: tensor<5x9xui32>) -> tensor<5x9xf16>
func.func @pointwise_mhlo_ops_non_divisible(%arg0: tensor<5x9xf32>, %arg1: tensor<5x9xui32>) -> tensor<5x9xf16> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<5x9xf32>, tensor<f32>) -> tensor<6x10xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 5, 0, 5]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[OFF1]], sizes = [3, 5] : (tensor<6x10xf32>, tensor<i32>, tensor<i32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0> : tensor<ui32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<5x9xui32>, tensor<ui32>) -> tensor<6x10xui32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 5, 0, 5]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF2]], %[[OFF3]], sizes = [3, 5] : (tensor<6x10xui32>, tensor<i32>, tensor<i32>) -> tensor<3x5xui32>
  // CHECK-NEXT: %[[ERF:.*]] = mhlo.erf %[[SLICE0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<3x5xf32>
  // CHECK-NEXT: %[[ACOS:.*]] = mhlo.acos %[[ERF]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<3x5xf32>
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %[[ACOS]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<3x5xf32>
  // CHECK-NEXT: %[[SC:.*]] = "mhlo.stochastic_convert"(%[[COPY]], %[[SLICE1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : (tensor<3x5xf32>, tensor<3x5xui32>) -> tensor<3x5xf16>
  // CHECK-NEXT: %[[AG_IN:.*]] = stablehlo.reshape %[[SC]] : (tensor<3x5xf16>) -> tensor<1x3x5xf16>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AG_IN]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<1x3x5xf16>) -> tensor<4x3x5xf16>
  // CHECK-NEXT: %[[AG_R0:.*]] = stablehlo.reshape %[[AG]] : (tensor<4x3x5xf16>) -> tensor<2x2x3x5xf16>
  // CHECK-NEXT: %[[AG_T:.*]] = stablehlo.transpose %[[AG_R0]], dims = [0, 2, 1, 3] : (tensor<2x2x3x5xf16>) -> tensor<2x3x2x5xf16>
  // CHECK-NEXT: %[[AG_R1:.*]] = stablehlo.reshape %[[AG_T]] : (tensor<2x3x2x5xf16>) -> tensor<6x10xf16>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG_R1]] [0:5, 0:9] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : (tensor<6x10xf16>) -> tensor<5x9xf16>
  // CHECK-NEXT: return %[[RES]] : tensor<5x9xf16>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {"y"}]> : tensor<5x9xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {"y"}]> : tensor<5x9xui32>
  %2 = mhlo.erf %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<5x9xf32>
  %3 = mhlo.acos %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<5x9xf32>
  %4 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : tensor<5x9xf32>
  %5 = "mhlo.stochastic_convert"(%4, %1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>} : (tensor<5x9xf32>, tensor<5x9xui32>) -> tensor<5x9xf16>
  %6 = sdy.reshard %5 <@mesh_xy, [{}, {}]> : tensor<5x9xf16>
  return %6 : tensor<5x9xf16>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @topk_non_divisible_batch
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x10xf32>) -> (tensor<5x3xf32>, tensor<5x3xi32>)
func.func @topk_non_divisible_batch(%arg0: tensor<5x10xf32>) -> (tensor<5x3xf32>, tensor<5x3xi32>) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<5x10xf32>, tensor<f32>) -> tensor<6x10xf32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF_S:.*]] = stablehlo.dynamic_slice %[[OFF_TBL]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[OFF0]], %[[OFF1]], sizes = [3, 10] : (tensor<6x10xf32>, tensor<i32>, tensor<i32>) -> tensor<3x10xf32>
  // CHECK-NEXT: %[[VALUES:.*]], %[[INDICES:.*]] = mhlo.topk(%[[SLICE]], k = 3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>, <@mesh_xy, [{"x"}, {}]>]>} : tensor<3x10xf32> -> (tensor<3x3xf32>, tensor<3x3xi32>)
  // CHECK-NEXT: %[[AG_V:.*]] = "stablehlo.all_gather"(%[[VALUES]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<3x3xf32>) -> tensor<6x3xf32>
  // CHECK-NEXT: %[[RES_V:.*]] = stablehlo.slice %[[AG_V]] [0:5, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : (tensor<6x3xf32>) -> tensor<5x3xf32>
  // CHECK-NEXT: %[[AG_I:.*]] = "stablehlo.all_gather"(%[[INDICES]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<3x3xi32>) -> tensor<6x3xi32>
  // CHECK-NEXT: %[[RES_I:.*]] = stablehlo.slice %[[AG_I]] [0:5, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : (tensor<6x3xi32>) -> tensor<5x3xi32>
  // CHECK-NEXT: return %[[RES_V]], %[[RES_I]] : tensor<5x3xf32>, tensor<5x3xi32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}]> : tensor<5x10xf32>
  %values, %indices = mhlo.topk(%0, k = 3, largest = true) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>, <@mesh_xy, [{"x"}, {}]>]>} : tensor<5x10xf32> -> (tensor<5x3xf32>, tensor<5x3xi32>)
  %v = sdy.reshard %values <@mesh_xy, [{}, {}]> : tensor<5x3xf32>
  %i = sdy.reshard %indices <@mesh_xy, [{}, {}]> : tensor<5x3xi32>
  return %v, %i : tensor<5x3xf32>, tensor<5x3xi32>
}

// -----

sdy.mesh @mesh_x = <["x"=2]>

// CHECK-LABEL: func @topk_k_equals_last_dim_non_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xi32>)
func.func @topk_k_equals_last_dim_non_divisible(%arg0: tensor<4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xi32>) {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x6xf32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF_TBL:.*]] = stablehlo.constant dense<[0, 3]> : tensor<2xi32>
  // CHECK-NEXT: %[[OFF_S:.*]] = stablehlo.dynamic_slice %[[OFF_TBL]], %[[PID]], sizes = [1] : (tensor<2xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[OFF1]], sizes = [4, 3] : (tensor<4x6xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[A2A_IN:.*]] = "stablehlo.all_to_all"(%[[SLICE0]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_x, axes = [#stablehlo.axis_ref<name = "x">]>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<4x3xf32>) -> tensor<2x6xf32>
  // CHECK-NEXT: %[[TOPK_IN:.*]] = stablehlo.slice %[[A2A_IN]] [0:2, 0:5] : (tensor<2x6xf32>) -> tensor<2x5xf32>
  // CHECK-NEXT: %[[VALUES:.*]], %[[INDICES:.*]] = mhlo.topk(%[[TOPK_IN]], k = 5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{"x"}, {}]>]>} : tensor<2x5xf32> -> (tensor<2x5xf32>, tensor<2x5xi32>)
  // CHECK-NEXT: %[[CST_V:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD_V:.*]] = stablehlo.pad %[[VALUES]], %[[CST_V]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<2x5xf32>, tensor<f32>) -> tensor<2x6xf32>
  // CHECK-NEXT: %[[A2A_V:.*]] = "stablehlo.all_to_all"(%[[PAD_V]]) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_x, axes = [#stablehlo.axis_ref<name = "x">]>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x6xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[CST_I:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD_I:.*]] = stablehlo.pad %[[INDICES]], %[[CST_I]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<2x5xi32>, tensor<i32>) -> tensor<2x6xi32>
  // CHECK-NEXT: %[[A2A_I:.*]] = "stablehlo.all_to_all"(%[[PAD_I]]) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, concat_dimension = 0 : i64, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_x, axes = [#stablehlo.axis_ref<name = "x">]>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x6xi32>) -> tensor<4x3xi32>
  // CHECK-NEXT: %[[AG_V:.*]] = "stablehlo.all_gather"(%[[A2A_V]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_x, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<4x3xf32>) -> tensor<4x6xf32>
  // CHECK-NEXT: %[[RES_V:.*]] = stablehlo.slice %[[AG_V]] [0:4, 0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{}, {}]>]>} : (tensor<4x6xf32>) -> tensor<4x5xf32>
  // CHECK-NEXT: %[[AG_I:.*]] = "stablehlo.all_gather"(%[[A2A_I]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_x, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<4x3xi32>) -> tensor<4x6xi32>
  // CHECK-NEXT: %[[RES_I:.*]] = stablehlo.slice %[[AG_I]] [0:4, 0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{}, {}]>]>} : (tensor<4x6xi32>) -> tensor<4x5xi32>
  // CHECK-NEXT: return %[[RES_V]], %[[RES_I]] : tensor<4x5xf32>, tensor<4x5xi32>
  %0 = sdy.reshard %arg0 <@mesh_x, [{}, {"x"}]> : tensor<4x5xf32>
  %values, %indices = mhlo.topk(%0, k = 5, largest = true) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{}, {"x"}]>, <@mesh_x, [{}, {"x"}]>]>} : tensor<4x5xf32> -> (tensor<4x5xf32>, tensor<4x5xi32>)
  %v = sdy.reshard %values <@mesh_x, [{}, {}]> : tensor<4x5xf32>
  %i = sdy.reshard %indices <@mesh_x, [{}, {}]> : tensor<4x5xi32>
  return %v, %i : tensor<4x5xf32>, tensor<4x5xi32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_non_divisible_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8x4xf32>, %[[ARG1:.*]]: tensor<2x4x4x7xf32>, %[[ARG2:.*]]: tensor<4x2xi32>) -> tensor<4x8x7xf32>
func.func @ragged_dot_non_divisible_output(
    %arg0: tensor<4x8x4xf32>,
    %arg1: tensor<2x4x4x7xf32>,
    %arg2: tensor<4x2xi32>) -> tensor<4x8x7xf32> {
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 2, 2]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[Z0]], %[[Z1]], sizes = [2, 8, 4] : (tensor<4x8x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x8x4xf32>
  // CHECK-NEXT: %[[CST_PAD:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RHS_PAD:.*]] = stablehlo.pad %[[ARG1]], %[[CST_PAD]], low = [0, 0, 0, 0], high = [0, 0, 0, 1], interior = [0, 0, 0, 0] : (tensor<2x4x4x7xf32>, tensor<f32>) -> tensor<2x4x4x8xf32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 2, 2]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[RHS_PAD]], %[[Z2]], %[[OFF1]], %[[Z3]], %[[OFF2]], sizes = [2, 2, 4, 4] : (tensor<2x4x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x2x4x4xf32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 0, 2, 2]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[GS:.*]] = stablehlo.dynamic_slice %[[ARG2]], %[[OFF3]], %[[Z4]], sizes = [2, 2] : (tensor<4x2xi32>, tensor<i32>, tensor<i32>) -> tensor<2x2xi32>
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[LHS]], %[[RHS]], %[[GS]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>} : (tensor<2x8x4xf32>, tensor<2x2x4x4xf32>, tensor<2x2xi32>) -> tensor<2x8x4xf32>
  // CHECK-NEXT: %[[AG_IN:.*]] = stablehlo.reshape %[[RDOT]] : (tensor<2x8x4xf32>) -> tensor<1x2x8x4xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AG_IN]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<1x2x8x4xf32>) -> tensor<4x2x8x4xf32>
  // CHECK-NEXT: %[[AG_R0:.*]] = stablehlo.reshape %[[AG]] : (tensor<4x2x8x4xf32>) -> tensor<2x2x2x8x4xf32>
  // CHECK-NEXT: %[[AG_T:.*]] = stablehlo.transpose %[[AG_R0]], dims = [0, 2, 3, 1, 4] : (tensor<2x2x2x8x4xf32>) -> tensor<2x2x8x2x4xf32>
  // CHECK-NEXT: %[[AG_R1:.*]] = stablehlo.reshape %[[AG_T]] : (tensor<2x2x8x2x4xf32>) -> tensor<4x8x8xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG_R1]] [0:4, 0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}]>]>} : (tensor<4x8x8xf32>) -> tensor<4x8x7xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x8x7xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {}]> : tensor<4x8x4xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{}, {"x"}, {}, {"y"}]> : tensor<2x4x4x7xf32>
  %2 = sdy.reshard %arg2 <@mesh_xy, [{"x"}, {}]> : tensor<4x2xi32>
  %3 = "mhlo.ragged_dot"(%0, %1, %2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>}
    : (tensor<4x8x4xf32>, tensor<2x4x4x7xf32>, tensor<4x2xi32>) -> tensor<4x8x7xf32>
  %4 = sdy.reshard %3 <@mesh_xy, [{}, {}, {}]> : tensor<4x8x7xf32>
  return %4 : tensor<4x8x7xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_non_divisible_batch_and_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x8x4xf32>, %[[ARG1:.*]]: tensor<2x5x4x7xf32>, %[[ARG2:.*]]: tensor<5x2xi32>) -> tensor<5x8x7xf32>
func.func @ragged_dot_non_divisible_batch_and_output(
    %arg0: tensor<5x8x4xf32>,
    %arg1: tensor<2x5x4x7xf32>,
    %arg2: tensor<5x2xi32>) -> tensor<5x8x7xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<5x8x4xf32>, tensor<f32>) -> tensor<6x8x4xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[Z0]], %[[Z1]], sizes = [3, 8, 4] : (tensor<6x8x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x8x4xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [0, 1, 0, 1], interior = [0, 0, 0, 0] : (tensor<2x5x4x7xf32>, tensor<f32>) -> tensor<2x6x4x8xf32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[Z2]], %[[OFF1]], %[[Z3]], %[[OFF2]], sizes = [2, 3, 4, 4] : (tensor<2x6x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x4x4xf32>
  // CHECK-NEXT: %[[CST2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD2:.*]] = stablehlo.pad %[[ARG2]], %[[CST2]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<5x2xi32>, tensor<i32>) -> tensor<6x2xi32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[GS:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[OFF3]], %[[Z4]], sizes = [3, 2] : (tensor<6x2xi32>, tensor<i32>, tensor<i32>) -> tensor<3x2xi32>
  // CHECK-NEXT: %[[GS_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<3x2xi32>
  // CHECK-NEXT: %[[PID4:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF4_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF4_S:.*]] = stablehlo.dynamic_slice %[[OFF4_TBL]], %[[PID4]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF4:.*]] = stablehlo.reshape %[[OFF4_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF4_CVT:.*]] = stablehlo.convert %[[OFF4]] : tensor<i32>
  // CHECK-NEXT: %[[OFF4_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF4_CVT]], dims = [] : (tensor<i32>) -> tensor<3x2xi32>
  // CHECK-NEXT: %[[GS_IDX:.*]] = stablehlo.add %[[GS_IOTA]], %[[OFF4_BCAST]] : tensor<3x2xi32>
  // CHECK-NEXT: %[[GS_LIMIT:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT: %[[GS_BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[GS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<3x2xi32>
  // CHECK-NEXT: %[[GS_CMP:.*]] = stablehlo.compare LT, %[[GS_IDX]], %[[GS_BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>]>} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi1>
  // CHECK-NEXT: %[[GS_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[GS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[GS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<3x2xi32>
  // CHECK-NEXT: %[[GS_MASKED:.*]] = stablehlo.select %[[GS_CMP]], %[[GS]], %[[GS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}]>]>} : tensor<3x2xi1>, tensor<3x2xi32>
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[LHS]], %[[RHS]], %[[GS_MASKED]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>} : (tensor<3x8x4xf32>, tensor<2x3x4x4xf32>, tensor<3x2xi32>) -> tensor<3x8x4xf32>
  // CHECK-NEXT: %[[AG_IN:.*]] = stablehlo.reshape %[[RDOT]] : (tensor<3x8x4xf32>) -> tensor<1x3x8x4xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AG_IN]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<1x3x8x4xf32>) -> tensor<4x3x8x4xf32>
  // CHECK-NEXT: %[[AG_R0:.*]] = stablehlo.reshape %[[AG]] : (tensor<4x3x8x4xf32>) -> tensor<2x2x3x8x4xf32>
  // CHECK-NEXT: %[[AG_T:.*]] = stablehlo.transpose %[[AG_R0]], dims = [0, 2, 3, 1, 4] : (tensor<2x2x3x8x4xf32>) -> tensor<2x3x8x2x4xf32>
  // CHECK-NEXT: %[[AG_R1:.*]] = stablehlo.reshape %[[AG_T]] : (tensor<2x3x8x2x4xf32>) -> tensor<6x8x8xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG_R1]] [0:5, 0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}]>]>} : (tensor<6x8x8xf32>) -> tensor<5x8x7xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<5x8x7xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {}]> : tensor<5x8x4xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{}, {"x"}, {}, {"y"}]> : tensor<2x5x4x7xf32>
  %2 = sdy.reshard %arg2 <@mesh_xy, [{"x"}, {}]> : tensor<5x2xi32>
  %3 = "mhlo.ragged_dot"(%0, %1, %2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>}
    : (tensor<5x8x4xf32>, tensor<2x5x4x7xf32>, tensor<5x2xi32>) -> tensor<5x8x7xf32>
  %4 = sdy.reshard %3 <@mesh_xy, [{}, {}, {}]> : tensor<5x8x7xf32>
  return %4 : tensor<5x8x7xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_non_divisible_contracting
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8x7xf32>, %[[ARG1:.*]]: tensor<2x4x7x8xf32>, %[[ARG2:.*]]: tensor<4x2xi32>) -> tensor<4x8x8xf32>
func.func @ragged_dot_non_divisible_contracting(
    %arg0: tensor<4x8x7xf32>,
    %arg1: tensor<2x4x7x8xf32>,
    %arg2: tensor<4x2xi32>) -> tensor<4x8x8xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<4x8x7xf32>, tensor<f32>) -> tensor<4x8x8xf32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[Z0]], %[[Z1]], %[[OFF0]], sizes = [4, 8, 4] : (tensor<4x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x4xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<2x4x7x8xf32>, tensor<f32>) -> tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[Z2]], %[[Z3]], %[[OFF1]], %[[Z4]], sizes = [2, 4, 4, 8] : (tensor<2x4x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4x8xf32>
  // CHECK-NEXT: %[[LHS_IOTA:.*]] = stablehlo.iota dim = 2 : tensor<4x8x4xi32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF2_CVT:.*]] = stablehlo.convert %[[OFF2]] : tensor<i32>
  // CHECK-NEXT: %[[OFF2_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF2_CVT]], dims = [] : (tensor<i32>) -> tensor<4x8x4xi32>
  // CHECK-NEXT: %[[LHS_IDX:.*]] = stablehlo.add %[[LHS_IOTA]], %[[OFF2_BCAST]] : tensor<4x8x4xi32>
  // CHECK-NEXT: %[[LHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[LHS_BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}]>]>} : (tensor<i32>) -> tensor<4x8x4xi32>
  // CHECK-NEXT: %[[LHS_CMP:.*]] = stablehlo.compare LT, %[[LHS_IDX]], %[[LHS_BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}]>]>} : (tensor<4x8x4xi32>, tensor<4x8x4xi32>) -> tensor<4x8x4xi1>
  // CHECK-NEXT: %[[LHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[LHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[LHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}]>]>} : (tensor<f32>) -> tensor<4x8x4xf32>
  // CHECK-NEXT: %[[LHS_MASKED:.*]] = stablehlo.select %[[LHS_CMP]], %[[LHS]], %[[LHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}]>]>} : tensor<4x8x4xi1>, tensor<4x8x4xf32>
  // CHECK-NEXT: %[[RHS_IOTA:.*]] = stablehlo.iota dim = 2 : tensor<2x4x4x8xi32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 0, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF3_CVT:.*]] = stablehlo.convert %[[OFF3]] : tensor<i32>
  // CHECK-NEXT: %[[OFF3_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF3_CVT]], dims = [] : (tensor<i32>) -> tensor<2x4x4x8xi32>
  // CHECK-NEXT: %[[RHS_IDX:.*]] = stablehlo.add %[[RHS_IOTA]], %[[OFF3_BCAST]] : tensor<2x4x4x8xi32>
  // CHECK-NEXT: %[[RHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[RHS_BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[RHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}, {}]>]>} : (tensor<i32>) -> tensor<2x4x4x8xi32>
  // CHECK-NEXT: %[[RHS_CMP:.*]] = stablehlo.compare LT, %[[RHS_IDX]], %[[RHS_BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}, {}]>]>} : (tensor<2x4x4x8xi32>, tensor<2x4x4x8xi32>) -> tensor<2x4x4x8xi1>
  // CHECK-NEXT: %[[RHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}, {}]>]>} : (tensor<f32>) -> tensor<2x4x4x8xf32>
  // CHECK-NEXT: %[[RHS_MASKED:.*]] = stablehlo.select %[[RHS_CMP]], %[[RHS]], %[[RHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {"x"}, {}]>]>} : tensor<2x4x4x8xi1>, tensor<2x4x4x8xf32>
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[LHS_MASKED]], %[[RHS_MASKED]], %[[ARG2]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}]>]>} : (tensor<4x8x4xf32>, tensor<2x4x4x8xf32>, tensor<4x2xi32>) -> tensor<4x8x8xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[RDOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[A0]], %[[A1]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<4x8x8xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{}, {}, {"x"}]> : tensor<4x8x7xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{}, {}, {"x"}, {}]> : tensor<2x4x7x8xf32>
  %2 = "mhlo.ragged_dot"(%0, %1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}]>]>}
    : (tensor<4x8x7xf32>, tensor<2x4x7x8xf32>, tensor<4x2xi32>) -> tensor<4x8x8xf32>
  return %2 : tensor<4x8x8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_1d_group_sizes_non_divisible_batch
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x8x4xf32>, %[[ARG1:.*]]: tensor<2x5x4x7xf32>, %[[ARG2:.*]]: tensor<2xi32>) -> tensor<5x8x7xf32>
func.func @ragged_dot_1d_group_sizes_non_divisible_batch(
    %arg0: tensor<5x8x4xf32>,
    %arg1: tensor<2x5x4x7xf32>,
    %arg2: tensor<2xi32>) -> tensor<5x8x7xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<5x8x4xf32>, tensor<f32>) -> tensor<6x8x4xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[Z0]], %[[Z1]], sizes = [3, 8, 4] : (tensor<6x8x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x8x4xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [0, 1, 0, 1], interior = [0, 0, 0, 0] : (tensor<2x5x4x7xf32>, tensor<f32>) -> tensor<2x6x4x8xf32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[Z2]], %[[OFF1]], %[[Z3]], %[[OFF2]], sizes = [2, 3, 4, 4] : (tensor<2x6x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x4x4xf32>
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[LHS]], %[[RHS]], %[[ARG2]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>} : (tensor<3x8x4xf32>, tensor<2x3x4x4xf32>, tensor<2xi32>) -> tensor<3x8x4xf32>
  // CHECK-NEXT: %[[AG_IN:.*]] = stablehlo.reshape %[[RDOT]] : (tensor<3x8x4xf32>) -> tensor<1x3x8x4xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AG_IN]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<1x3x8x4xf32>) -> tensor<4x3x8x4xf32>
  // CHECK-NEXT: %[[AG_R0:.*]] = stablehlo.reshape %[[AG]] : (tensor<4x3x8x4xf32>) -> tensor<2x2x3x8x4xf32>
  // CHECK-NEXT: %[[AG_T:.*]] = stablehlo.transpose %[[AG_R0]], dims = [0, 2, 3, 1, 4] : (tensor<2x2x3x8x4xf32>) -> tensor<2x3x8x2x4xf32>
  // CHECK-NEXT: %[[AG_R1:.*]] = stablehlo.reshape %[[AG_T]] : (tensor<2x3x8x2x4xf32>) -> tensor<6x8x8xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG_R1]] [0:5, 0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}]>]>} : (tensor<6x8x8xf32>) -> tensor<5x8x7xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<5x8x7xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {}]> : tensor<5x8x4xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{}, {"x"}, {}, {"y"}]> : tensor<2x5x4x7xf32>
  %2 = "mhlo.ragged_dot"(%0, %1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}]>]>}
    : (tensor<5x8x4xf32>, tensor<2x5x4x7xf32>, tensor<2xi32>) -> tensor<5x8x7xf32>
  %3 = sdy.reshard %2 <@mesh_xy, [{}, {}, {}]> : tensor<5x8x7xf32>
  return %3 : tensor<5x8x7xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_contracting_mode_non_divisible_batch_and_contracting
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x4x7x8xf32>, %[[ARG1:.*]]: tensor<5x7x8x6xf32>, %[[ARG2:.*]]: tensor<5x7x2xi32>) -> tensor<2x5x4x6xf32>
func.func @ragged_dot_contracting_mode_non_divisible_batch_and_contracting(
    %arg0: tensor<5x4x7x8xf32>,
    %arg1: tensor<5x7x8x6xf32>,
    %arg2: tensor<5x7x2xi32>) -> tensor<2x5x4x6xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0, 0], high = [1, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<5x4x7x8xf32>, tensor<f32>) -> tensor<6x4x8x8xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[Z0]], %[[OFF1]], %[[Z1]], sizes = [3, 4, 4, 8] : (tensor<6x4x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x4x8xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [1, 1, 0, 0], interior = [0, 0, 0, 0] : (tensor<5x7x8x6xf32>, tensor<f32>) -> tensor<6x8x8x6xf32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF2]], %[[OFF3]], %[[Z2]], %[[Z3]], sizes = [3, 4, 8, 6] : (tensor<6x8x8x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[CST2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD2:.*]] = stablehlo.pad %[[ARG2]], %[[CST2]], low = [0, 0, 0], high = [1, 1, 0], interior = [0, 0, 0] : (tensor<5x7x2xi32>, tensor<i32>) -> tensor<6x8x2xi32>
  // CHECK-NEXT: %[[PID4:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF4_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF4_S:.*]] = stablehlo.dynamic_slice %[[OFF4_TBL]], %[[PID4]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF4:.*]] = stablehlo.reshape %[[OFF4_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID5:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF5_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF5_S:.*]] = stablehlo.dynamic_slice %[[OFF5_TBL]], %[[PID5]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF5:.*]] = stablehlo.reshape %[[OFF5_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[GS:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[OFF4]], %[[OFF5]], %[[Z4]], sizes = [3, 4, 2] : (tensor<6x8x2xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[LHS_IOTA:.*]] = stablehlo.iota dim = 2 : tensor<3x4x4x8xi32>
  // CHECK-NEXT: %[[PID6:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF6_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF6_S:.*]] = stablehlo.dynamic_slice %[[OFF6_TBL]], %[[PID6]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF6:.*]] = stablehlo.reshape %[[OFF6_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF6_CVT:.*]] = stablehlo.convert %[[OFF6]] : tensor<i32>
  // CHECK-NEXT: %[[OFF6_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF6_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x4x8xi32>
  // CHECK-NEXT: %[[LHS_IDX:.*]] = stablehlo.add %[[LHS_IOTA]], %[[OFF6_BCAST]] : tensor<3x4x4x8xi32>
  // CHECK-NEXT: %[[LHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[LHS_BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<i32>) -> tensor<3x4x4x8xi32>
  // CHECK-NEXT: %[[LHS_CMP:.*]] = stablehlo.compare LT, %[[LHS_IDX]], %[[LHS_BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<3x4x4x8xi32>, tensor<3x4x4x8xi32>) -> tensor<3x4x4x8xi1>
  // CHECK-NEXT: %[[LHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[LHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[LHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<f32>) -> tensor<3x4x4x8xf32>
  // CHECK-NEXT: %[[LHS_MASKED:.*]] = stablehlo.select %[[LHS_CMP]], %[[LHS]], %[[LHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : tensor<3x4x4x8xi1>, tensor<3x4x4x8xf32>
  // CHECK-NEXT: %[[RHS_IOTA:.*]] = stablehlo.iota dim = 1 : tensor<3x4x8x6xi32>
  // CHECK-NEXT: %[[PID7:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF7_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF7_S:.*]] = stablehlo.dynamic_slice %[[OFF7_TBL]], %[[PID7]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF7:.*]] = stablehlo.reshape %[[OFF7_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF7_CVT:.*]] = stablehlo.convert %[[OFF7]] : tensor<i32>
  // CHECK-NEXT: %[[OFF7_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF7_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x8x6xi32>
  // CHECK-NEXT: %[[RHS_IDX:.*]] = stablehlo.add %[[RHS_IOTA]], %[[OFF7_BCAST]] : tensor<3x4x8x6xi32>
  // CHECK-NEXT: %[[RHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[RHS_BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[RHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<i32>) -> tensor<3x4x8x6xi32>
  // CHECK-NEXT: %[[RHS_CMP:.*]] = stablehlo.compare LT, %[[RHS_IDX]], %[[RHS_BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<3x4x8x6xi32>, tensor<3x4x8x6xi32>) -> tensor<3x4x8x6xi1>
  // CHECK-NEXT: %[[RHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<f32>) -> tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[RHS_MASKED:.*]] = stablehlo.select %[[RHS_CMP]], %[[RHS]], %[[RHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}, {}]>]>} : tensor<3x4x8x6xi1>, tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[GS_IOTA0:.*]] = stablehlo.iota dim = 0 : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[PID8:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF8_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF8_S:.*]] = stablehlo.dynamic_slice %[[OFF8_TBL]], %[[PID8]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF8:.*]] = stablehlo.reshape %[[OFF8_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF8_CVT:.*]] = stablehlo.convert %[[OFF8]] : tensor<i32>
  // CHECK-NEXT: %[[OFF8_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF8_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_IDX0:.*]] = stablehlo.add %[[GS_IOTA0]], %[[OFF8_BCAST]] : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_LIMIT0:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT: %[[GS_BCAST0:.*]] = stablehlo.broadcast_in_dim %[[GS_LIMIT0]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : (tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_CMP0:.*]] = stablehlo.compare LT, %[[GS_IDX0]], %[[GS_BCAST0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : (tensor<3x4x2xi32>, tensor<3x4x2xi32>) -> tensor<3x4x2xi1>
  // CHECK-NEXT: %[[GS_IOTA1:.*]] = stablehlo.iota dim = 1 : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[PID9:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF9_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF9_S:.*]] = stablehlo.dynamic_slice %[[OFF9_TBL]], %[[PID9]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF9:.*]] = stablehlo.reshape %[[OFF9_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF9_CVT:.*]] = stablehlo.convert %[[OFF9]] : tensor<i32>
  // CHECK-NEXT: %[[OFF9_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF9_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_IDX1:.*]] = stablehlo.add %[[GS_IOTA1]], %[[OFF9_BCAST]] : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_LIMIT1:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[GS_BCAST1:.*]] = stablehlo.broadcast_in_dim %[[GS_LIMIT1]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : (tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_CMP1:.*]] = stablehlo.compare LT, %[[GS_IDX1]], %[[GS_BCAST1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : (tensor<3x4x2xi32>, tensor<3x4x2xi32>) -> tensor<3x4x2xi1>
  // CHECK-NEXT: %[[GS_AND:.*]] = stablehlo.and %[[GS_CMP0]], %[[GS_CMP1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : tensor<3x4x2xi1>
  // CHECK-NEXT: %[[GS_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[GS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[GS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : (tensor<i32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[GS_MASKED:.*]] = stablehlo.select %[[GS_AND]], %[[GS]], %[[GS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}, {}]>]>} : tensor<3x4x2xi1>, tensor<3x4x2xi32>
  // CHECK-NEXT: %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[LHS_MASKED]], %[[RHS_MASKED]], %[[GS_MASKED]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 3], rhs_contracting_dimensions = [1, 2]>, lhs_ragged_dimensions = [3]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}, {}]>]>} : (tensor<3x4x4x8xf32>, tensor<3x4x8x6xf32>, tensor<3x4x2xi32>) -> tensor<2x3x4x6xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[RDOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[A0]], %[[A1]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<2x3x4x6xf32>) -> tensor<2x3x4x6xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AR]]) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<2x3x4x6xf32>) -> tensor<2x6x4x6xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG]] [0:2, 0:5, 0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}, {}]>]>} : (tensor<2x6x4x6xf32>) -> tensor<2x5x4x6xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x5x4x6xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {"y"}, {}]> : tensor<5x4x7x8xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {"y"}, {}, {}]> : tensor<5x7x8x6xf32>
  %2 = sdy.reshard %arg2 <@mesh_xy, [{"x"}, {"y"}, {}]> : tensor<5x7x2xi32>
  %3 = "mhlo.ragged_dot"(%0, %1, %2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2, 3], rhs_contracting_dimensions = [1, 2]>,
      lhs_ragged_dimensions = [3], rhs_group_dimensions = []>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}, {}]>]>}
    : (tensor<5x4x7x8xf32>, tensor<5x7x8x6xf32>, tensor<5x7x2xi32>) -> tensor<2x5x4x6xf32>
  %4 = sdy.reshard %3 <@mesh_xy, [{}, {}, {}, {}]> : tensor<2x5x4x6xf32>
  return %4 : tensor<2x5x4x6xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_batch_mode_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8x6xf32>, %[[ARG1:.*]]: tensor<4x6x10xf32>, %[[ARG2:.*]]: tensor<2xi32>) -> tensor<4x8x10xf32>
func.func @ragged_dot_batch_mode_divisible(
    %arg0: tensor<4x8x6xf32>,
    %arg1: tensor<4x6x10xf32>,
    %arg2: tensor<2xi32>) -> tensor<4x8x10xf32> {
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 2, 2]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 3, 0, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[Z0]], %[[OFF1]], sizes = [2, 8, 3] : (tensor<4x8x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x8x3xf32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 2, 2]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 3, 0, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[ARG1]], %[[OFF2]], %[[OFF3]], %[[Z1]], sizes = [2, 3, 10] : (tensor<4x6x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x10xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[LHS]], %[[RHS]], batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGH, HIGHEST] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}]>]>} : (tensor<2x8x3xf32>, tensor<2x3x10xf32>) -> tensor<2x8x10xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[DOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[A0]], %[[A1]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<2x8x10xf32>) -> tensor<2x8x10xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AR]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<2x8x10xf32>) -> tensor<4x8x10xf32>
  // CHECK-NEXT: return %[[AG]] : tensor<4x8x10xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {"y"}]> : tensor<4x8x6xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {"y"}, {}]> : tensor<4x6x10xf32>
  %2 = "mhlo.ragged_dot"(%0, %1, %arg2) <{
    precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGHEST>],
    ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      lhs_ragged_dimensions = [0], rhs_group_dimensions = []>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}]>]>}
    : (tensor<4x8x6xf32>, tensor<4x6x10xf32>, tensor<2xi32>) -> tensor<4x8x10xf32>
  %3 = sdy.reshard %2 <@mesh_xy, [{}, {}, {}]> : tensor<4x8x10xf32>
  return %3 : tensor<4x8x10xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_batch_mode_non_divisible_rank2_group_sizes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x4x8x7xf32>, %[[ARG1:.*]]: tensor<5x4x7x6xf32>, %[[ARG2:.*]]: tensor<5x2xi32>) -> tensor<5x4x8x6xf32>
func.func @ragged_dot_batch_mode_non_divisible_rank2_group_sizes(
    %arg0: tensor<5x4x8x7xf32>,
    %arg1: tensor<5x4x7x6xf32>,
    %arg2: tensor<5x2xi32>) -> tensor<5x4x8x6xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0, 0], high = [1, 0, 0, 1], interior = [0, 0, 0, 0] : (tensor<5x4x8x7xf32>, tensor<f32>) -> tensor<6x4x8x8xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[Z0]], %[[Z1]], %[[OFF1]], sizes = [3, 4, 8, 4] : (tensor<6x4x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x8x4xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [1, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<5x4x7x6xf32>, tensor<f32>) -> tensor<6x4x8x6xf32>
  // CHECK-NEXT: %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF2_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF2_S:.*]] = stablehlo.dynamic_slice %[[OFF2_TBL]], %[[PID2]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF2:.*]] = stablehlo.reshape %[[OFF2_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF3_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF3_S:.*]] = stablehlo.dynamic_slice %[[OFF3_TBL]], %[[PID3]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF3:.*]] = stablehlo.reshape %[[OFF3_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF2]], %[[Z2]], %[[OFF3]], %[[Z3]], sizes = [3, 4, 4, 6] : (tensor<6x4x8x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x4x6xf32>
  // CHECK-NEXT: %[[LHS_IOTA:.*]] = stablehlo.iota dim = 3 : tensor<3x4x8x4xi32>
  // CHECK-NEXT: %[[PID4:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF4_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF4_S:.*]] = stablehlo.dynamic_slice %[[OFF4_TBL]], %[[PID4]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF4:.*]] = stablehlo.reshape %[[OFF4_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF4_CVT:.*]] = stablehlo.convert %[[OFF4]] : tensor<i32>
  // CHECK-NEXT: %[[OFF4_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF4_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x8x4xi32>
  // CHECK-NEXT: %[[LHS_IDX:.*]] = stablehlo.add %[[LHS_IOTA]], %[[OFF4_BCAST]] : tensor<3x4x8x4xi32>
  // CHECK-NEXT: %[[LHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[LHS_BCAST:.*]] = stablehlo.broadcast_in_dim %[[LHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {"y"}]>]>} : (tensor<i32>) -> tensor<3x4x8x4xi32>
  // CHECK-NEXT: %[[LHS_CMP:.*]] = stablehlo.compare LT, %[[LHS_IDX]], %[[LHS_BCAST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {"y"}]>]>} : (tensor<3x4x8x4xi32>, tensor<3x4x8x4xi32>) -> tensor<3x4x8x4xi1>
  // CHECK-NEXT: %[[LHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[LHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[LHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {"y"}]>]>} : (tensor<f32>) -> tensor<3x4x8x4xf32>
  // CHECK-NEXT: %[[LHS_MASKED:.*]] = stablehlo.select %[[LHS_CMP]], %[[LHS]], %[[LHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {"y"}]>]>} : tensor<3x4x8x4xi1>, tensor<3x4x8x4xf32>
  // CHECK-NEXT: %[[RHS_IOTA:.*]] = stablehlo.iota dim = 2 : tensor<3x4x4x6xi32>
  // CHECK-NEXT: %[[PID5:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF5_TBL:.*]] = stablehlo.constant dense<[0, 4, 0, 4]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF5_S:.*]] = stablehlo.dynamic_slice %[[OFF5_TBL]], %[[PID5]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF5:.*]] = stablehlo.reshape %[[OFF5_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF5_CVT:.*]] = stablehlo.convert %[[OFF5]] : tensor<i32>
  // CHECK-NEXT: %[[OFF5_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFF5_CVT]], dims = [] : (tensor<i32>) -> tensor<3x4x4x6xi32>
  // CHECK-NEXT: %[[RHS_IDX:.*]] = stablehlo.add %[[RHS_IOTA]], %[[OFF5_BCAST]] : tensor<3x4x4x6xi32>
  // CHECK-NEXT: %[[RHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[RHS_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<i32>) -> tensor<3x4x4x6xi32>
  // CHECK-NEXT: %[[RHS_CMP:.*]] = stablehlo.compare LT, %[[RHS_IDX]], %[[RHS_BCAST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<3x4x4x6xi32>, tensor<3x4x4x6xi32>) -> tensor<3x4x4x6xi1>
  // CHECK-NEXT: %[[RHS_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RHS_BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZERO]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : (tensor<f32>) -> tensor<3x4x4x6xf32>
  // CHECK-NEXT: %[[RHS_MASKED:.*]] = stablehlo.select %[[RHS_CMP]], %[[RHS]], %[[RHS_BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {"y"}, {}]>]>} : tensor<3x4x4x6xi1>, tensor<3x4x4x6xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[LHS_MASKED]], %[[RHS_MASKED]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, HIGH] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {}]>]>} : (tensor<3x4x8x4xf32>, tensor<3x4x4x6xf32>) -> tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[DOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%[[A0:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[A0]], %[[A1]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<3x4x8x6xf32>) -> tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[AR]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<3x4x8x6xf32>) -> tensor<6x4x8x6xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG]] [0:5, 0:4, 0:8, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}, {}]>]>} : (tensor<6x4x8x6xf32>) -> tensor<5x4x8x6xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<5x4x8x6xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {}, {"y"}]> : tensor<5x4x8x7xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {}, {"y"}, {}]> : tensor<5x4x7x6xf32>
  %2 = "mhlo.ragged_dot"(%0, %1, %arg2) <{
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision HIGH>],
    ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = []>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {}]>]>}
    : (tensor<5x4x8x7xf32>, tensor<5x4x7x6xf32>, tensor<5x2xi32>) -> tensor<5x4x8x6xf32>
  %3 = sdy.reshard %2 <@mesh_xy, [{}, {}, {}, {}]> : tensor<5x4x8x6xf32>
  return %3 : tensor<5x4x8x6xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_batch_mode_non_divisible_batch_only
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x4x8x6xf32>, %[[ARG1:.*]]: tensor<5x4x6x4xf32>, %[[ARG2:.*]]: tensor<5x2xi32>) -> tensor<5x4x8x4xf32>
func.func @ragged_dot_batch_mode_non_divisible_batch_only(
    %arg0: tensor<5x4x8x6xf32>,
    %arg1: tensor<5x4x6x4xf32>,
    %arg2: tensor<5x2xi32>) -> tensor<5x4x8x4xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0, 0, 0], high = [1, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<5x4x8x6xf32>, tensor<f32>) -> tensor<6x4x8x6xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LHS:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0]], %[[Z0]], %[[Z1]], %[[Z2]], sizes = [3, 4, 8, 6] : (tensor<6x4x8x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x8x6xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0, 0], high = [1, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<5x4x6x4xf32>, tensor<f32>) -> tensor<6x4x6x4xf32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[Z3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[Z5:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF1]], %[[Z3]], %[[Z4]], %[[Z5]], sizes = [3, 4, 6, 4] : (tensor<6x4x6x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x6x4xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[LHS]], %[[RHS]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {}]>]>} : (tensor<3x4x8x6xf32>, tensor<3x4x6x4xf32>) -> tensor<3x4x8x4xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[DOT]]) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<3x4x8x4xf32>) -> tensor<6x4x8x4xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.slice %[[AG]] [0:5, 0:4, 0:8, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}, {}, {}]>]>} : (tensor<6x4x8x4xf32>) -> tensor<5x4x8x4xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<5x4x8x4xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {}, {}, {}]> : tensor<5x4x8x6xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {}, {}, {}]> : tensor<5x4x6x4xf32>
  %2 = "mhlo.ragged_dot"(%0, %1, %arg2) <{
    ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = []>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}, {}]>]>}
    : (tensor<5x4x8x6xf32>, tensor<5x4x6x4xf32>, tensor<5x2xi32>) -> tensor<5x4x8x4xf32>
  %3 = sdy.reshard %2 <@mesh_xy, [{}, {}, {}, {}]> : tensor<5x4x8x4xf32>
  return %3 : tensor<5x4x8x4xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// CHECK-LABEL: func @ragged_dot_while_block_arg
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x5xf32>, %[[ARG1:.*]]: tensor<2x5x8xf32>, %[[ARG2:.*]]: tensor<2xi32>, %[[ARG3:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @ragged_dot_while_block_arg(
    %arg0: tensor<4x5xf32>,
    %arg1: tensor<2x5x8xf32>,
    %arg2: tensor<2xi32>,
    %arg3: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x6xf32>
  // CHECK-NEXT: %[[OFF0_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF0_S:.*]] = stablehlo.dynamic_slice %[[OFF0_TBL]], %[[PID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF0_1:.*]] = stablehlo.reshape %[[OFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[OFF0_0]], %[[OFF0_1]], sizes = [4, 3] : (tensor<4x6xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<2x5x8xf32>, tensor<f32>) -> tensor<2x6x8xf32>
  // CHECK-NEXT: %[[OFF1_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[OFF1_S:.*]] = stablehlo.dynamic_slice %[[OFF1_TBL]], %[[PID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFF1_1:.*]] = stablehlo.reshape %[[OFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFF1_2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OFF1_0]], %[[OFF1_1]], %[[OFF1_2]], sizes = [2, 3, 8] : (tensor<2x6x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x8xf32>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[WHILE:.*]]:3 = stablehlo.while(%[[ITER0:.*]] = %[[SLICE0]], %[[ITER1:.*]] = %[[ARG3]], %[[ITER2:.*]] = %[[C0]]) : tensor<4x3xf32>, tensor<4x8xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>, <@mesh_xy, [{}, {}]>, <@mesh_xy, []>]>}
  // CHECK-NEXT: cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %[[ITER2]], %[[C4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[COND]] : tensor<i1>
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[STEP:.*]] = stablehlo.add %[[ITER2]], %[[C1]] : tensor<i32>
  // CHECK-NEXT:   %[[IOTA0:.*]] = stablehlo.iota dim = 1 : tensor<4x3xi32>
  // CHECK-NEXT:   %[[MPID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT:   %[[MOFF0_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT:   %[[MOFF0_S:.*]] = stablehlo.dynamic_slice %[[MOFF0_TBL]], %[[MPID0]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT:   %[[MOFF0:.*]] = stablehlo.reshape %[[MOFF0_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT:   %[[MCVT0:.*]] = stablehlo.convert %[[MOFF0]] : tensor<i32>
  // CHECK-NEXT:   %[[MBCAST0:.*]] = stablehlo.broadcast_in_dim %[[MCVT0]], dims = [] : (tensor<i32>) -> tensor<4x3xi32>
  // CHECK-NEXT:   %[[MADD0:.*]] = stablehlo.add %[[IOTA0]], %[[MBCAST0]] : tensor<4x3xi32>
  // CHECK-NEXT:   %[[MLIM0:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT:   %[[MLIMB0:.*]] = stablehlo.broadcast_in_dim %[[MLIM0]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>]>} : (tensor<i32>) -> tensor<4x3xi32>
  // CHECK-NEXT:   %[[MCMP0:.*]] = stablehlo.compare LT, %[[MADD0]], %[[MLIMB0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>]>} : (tensor<4x3xi32>, tensor<4x3xi32>) -> tensor<4x3xi1>
  // CHECK-NEXT:   %[[MZERO0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT:   %[[MZEROB0:.*]] = stablehlo.broadcast_in_dim %[[MZERO0]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>]>} : (tensor<f32>) -> tensor<4x3xf32>
  // CHECK-NEXT:   %[[MASKED0:.*]] = stablehlo.select %[[MCMP0]], %[[ITER0]], %[[MZEROB0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>]>} : tensor<4x3xi1>, tensor<4x3xf32>
  // CHECK-NEXT:   %[[IOTA1:.*]] = stablehlo.iota dim = 1 : tensor<2x3x8xi32>
  // CHECK-NEXT:   %[[MPID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT:   %[[MOFF1_TBL:.*]] = stablehlo.constant dense<[0, 0, 3, 3]> : tensor<4xi32>
  // CHECK-NEXT:   %[[MOFF1_S:.*]] = stablehlo.dynamic_slice %[[MOFF1_TBL]], %[[MPID1]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT:   %[[MOFF1:.*]] = stablehlo.reshape %[[MOFF1_S]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT:   %[[MCVT1:.*]] = stablehlo.convert %[[MOFF1]] : tensor<i32>
  // CHECK-NEXT:   %[[MBCAST1:.*]] = stablehlo.broadcast_in_dim %[[MCVT1]], dims = [] : (tensor<i32>) -> tensor<2x3x8xi32>
  // CHECK-NEXT:   %[[MADD1:.*]] = stablehlo.add %[[IOTA1]], %[[MBCAST1]] : tensor<2x3x8xi32>
  // CHECK-NEXT:   %[[MLIM1:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT:   %[[MLIMB1:.*]] = stablehlo.broadcast_in_dim %[[MLIM1]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}]>]>} : (tensor<i32>) -> tensor<2x3x8xi32>
  // CHECK-NEXT:   %[[MCMP1:.*]] = stablehlo.compare LT, %[[MADD1]], %[[MLIMB1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}]>]>} : (tensor<2x3x8xi32>, tensor<2x3x8xi32>) -> tensor<2x3x8xi1>
  // CHECK-NEXT:   %[[MZERO1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT:   %[[MZEROB1:.*]] = stablehlo.broadcast_in_dim %[[MZERO1]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}]>]>} : (tensor<f32>) -> tensor<2x3x8xf32>
  // CHECK-NEXT:   %[[MASKED1:.*]] = stablehlo.select %[[MCMP1]], %[[SLICE1]], %[[MZEROB1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}, {}]>]>} : tensor<2x3x8xi1>, tensor<2x3x8xf32>
  // CHECK-NEXT:   %[[RDOT:.*]] = "mhlo.ragged_dot"(%[[MASKED0]], %[[MASKED1]], %[[ARG2]]) <{ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, lhs_ragged_dimensions = [0], rhs_group_dimensions = [0]>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : (tensor<4x3xf32>, tensor<2x3x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
  // CHECK-NEXT:   %[[AR:.*]] = "stablehlo.all_reduce"(%[[RDOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
  // CHECK-NEXT:   ^bb0(%[[AR_A:.*]]: tensor<f32>, %[[AR_B:.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[AR_ADD:.*]] = stablehlo.add %[[AR_A]], %[[AR_B]] : tensor<f32>
  // CHECK-NEXT:     stablehlo.return %[[AR_ADD]] : tensor<f32>
  // CHECK-NEXT:   }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:   %[[ACC:.*]] = stablehlo.add %[[ITER1]], %[[AR]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : tensor<4x8xf32>
  // CHECK-NEXT:   stablehlo.return %[[ITER0]], %[[ACC]], %[[STEP]] : tensor<4x3xf32>, tensor<4x8xf32>, tensor<i32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#1 : tensor<4x8xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{}, {"x"}]> : tensor<4x5xf32>
  %1 = sdy.reshard %arg1 <@mesh_xy, [{}, {"x"}, {}]> : tensor<2x5x8xf32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %2:3 = stablehlo.while(%iterArg = %0, %iterArg_0 = %arg3, %iterArg_1 = %c0) : tensor<4x5xf32>, tensor<4x8xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"x"}]>, <@mesh_xy, [{}, {}]>, <@mesh_xy, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iterArg_1, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %step = stablehlo.add %iterArg_1, %c1 : tensor<i32>
      %dot = "mhlo.ragged_dot"(%iterArg, %1, %arg2) <{
        ragged_dot_dimension_numbers = #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
          lhs_batching_dimensions = [], rhs_batching_dimensions = [],
          lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
          lhs_ragged_dimensions = [0], rhs_group_dimensions = [0]>}>
        {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>}
        : (tensor<4x5xf32>, tensor<2x5x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
      %acc = stablehlo.add %iterArg_0, %dot {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {}]>]>} : tensor<4x8xf32>
      stablehlo.return %iterArg, %acc, %step : tensor<4x5xf32>, tensor<4x8xf32>, tensor<i32>
    }
  return %2#1 : tensor<4x8xf32>
}
