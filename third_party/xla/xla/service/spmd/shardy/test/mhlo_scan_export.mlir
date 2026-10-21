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
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' -sdy-resolve-permutation-factors -sdy-reshard-to-collectives -sdy-convert-global-to-local -split-input-file | FileCheck %s

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// Partitions mhlo.scan along a divisible non-scan dimension ("y").
// CHECK-LABEL: func @scan_non_scan_dim_sharded
// CHECK-SAME: (%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>},
// CHECK-SAME:  %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
// CHECK-SAME: -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
func.func @scan_non_scan_dim_sharded(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<4xf32>, %[[CARRY:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[IN]], %[[CARRY]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]], %[[ADD]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<16x4xf32>, tensor<4xf32>) -> (tensor<16x4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %[[SCAN]]#0, %[[SCAN]]#1 : tensor<16x4xf32>, tensor<4xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"y"}]>, <@mesh_xy, [{"y"}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<16x8xf32>, tensor<8xf32>) -> (tensor<16x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<16x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// Preserves scan_dim_size and broadcasts scalars inside the scan body.
// CHECK-LABEL: func @scan_non_scan_dim_sharded_with_scan_dim_size_and_scalar_broadcast
// CHECK-SAME: (%arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>},
// CHECK-SAME:  %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
// CHECK-SAME: -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
func.func @scan_non_scan_dim_sharded_with_scan_dim_size_and_scalar_broadcast(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%arg1) dimension=1
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 16 : i64
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<4xf32>, %[[CARRY:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT:   %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST]], dims = [] : (tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[IN]], %[[BCAST]] : tensor<4xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[CARRY]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]], %[[ADD]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<4x16xf32>, tensor<4xf32>) -> (tensor<4x16xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %[[SCAN]]#0, %[[SCAN]]#1 : tensor<4x16xf32>, tensor<4xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=1 attributes {
      is_associative = true,
      scan_dim_size = 16 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"y"}, {}]>, <@mesh_xy, [{"y"}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %cst = stablehlo.constant dense<1.0> : tensor<f32>
    %b = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %input0, %b : tensor<8xf32>
    %2 = stablehlo.add %1, %carry0 : tensor<8xf32>
    stablehlo.return %2, %2 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<8x16xf32>, tensor<8xf32>) -> (tensor<8x16xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8x16xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// Replicates the scan dimension ("x") when is_associative = false while keeping the non-scan dimension ("y") sharded.
// CHECK-LABEL: func @scan_non_associative_replicates_scan_dim_and_shards_non_scan_dim
// CHECK-SAME: (%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>},
// CHECK-SAME:  %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
// CHECK-SAME: -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
func.func @scan_non_associative_replicates_scan_dim_and_shards_non_scan_dim(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {"y"}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> : (tensor<8x4xf32>) -> tensor<16x4xf32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%[[AG]]) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = false
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<4xf32>, %[[CARRY:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[SUB:.*]] = stablehlo.subtract %[[IN]], %[[CARRY]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[SUB]], %[[SUB]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<16x4xf32>, tensor<4xf32>) -> (tensor<16x4xf32>, tensor<4xf32>)
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[SCAN]]#0, %{{.*}}, %{{.*}}, sizes = [8, 4] : (tensor<16x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x4xf32>
  // CHECK-NEXT: return %[[DS]], %[[SCAN]]#1 : tensor<8x4xf32>, tensor<4xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = false,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>, <@mesh_xy, [{"y"}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.subtract %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<16x8xf32>, tensor<8xf32>) -> (tensor<16x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<16x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// Replicates the non-scan dimension ("y") when the scan body has non-elementwise ops (dot_general).
// CHECK-LABEL: func @scan_non_elementwise_body_replicates_non_scan_dim
// CHECK-SAME: (%arg0: tensor<16x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}, {}]>},
// CHECK-SAME:  %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>})
// CHECK-SAME: -> (tensor<16x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}, {}]>},
// CHECK-SAME:     tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>})
func.func @scan_non_elementwise_body_replicates_non_scan_dim(
    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>})
    -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}, {}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {}]>}) {
  // CHECK-NEXT: %[[AG0:.*]] = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<16x4x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: %[[AG1:.*]] = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_xy, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> : (tensor<4x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%[[AG0]]) inits (%[[AG1]]) dimension=0
  // CHECK-SAME:   is_associative = false
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<8x8xf32>, %[[CARRY:.*]]: tensor<8x8xf32>):
  // CHECK-NEXT:   %[[DOT:.*]] = stablehlo.dot_general %[[IN]], %[[CARRY]], contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:   stablehlo.return %[[DOT]], %[[DOT]] : tensor<8x8xf32>, tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<16x8x8xf32>, tensor<8x8xf32>) -> (tensor<16x8x8xf32>, tensor<8x8xf32>)
  // CHECK: %[[DS0:.*]] = stablehlo.dynamic_slice %[[SCAN]]#0, %{{.*}}, %{{.*}}, %{{.*}}, sizes = [16, 4, 8] : (tensor<16x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x4x8xf32>
  // CHECK: %[[DS1:.*]] = stablehlo.dynamic_slice %[[SCAN]]#1, %{{.*}}, %{{.*}}, sizes = [4, 8] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[DS0]], %[[DS1]] : tensor<16x4x8xf32>, tensor<4x8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = false,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{}, {"y"}, {}]>, <@mesh_xy, [{"y"}, {}]>]>
  } {
  ^bb0(%input0: tensor<8x8xf32>, %carry0: tensor<8x8xf32>):
    %1 = stablehlo.dot_general %input0, %carry0, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    stablehlo.return %1, %1 : tensor<8x8xf32>, tensor<8x8xf32>
  } : (tensor<16x8x8xf32>, tensor<8x8xf32>) -> (tensor<16x8x8xf32>, tensor<8x8xf32>)
  return %0#0, %0#1 : tensor<16x8x8xf32>, tensor<8x8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

// Leaves a replicated non-shape-preserving mhlo.scan unchanged.
// CHECK-LABEL: func @scan_non_shape_preserving_replicated_passes_through
// CHECK-SAME: (%arg0: tensor<16x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<16x8xf32>, tensor<8x8xf32>)
func.func @scan_non_shape_preserving_replicated_passes_through(
    %arg0: tensor<16x8xf32>, %arg1: tensor<8x8xf32>)
    -> (tensor<16x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = false
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<8xf32>, %[[CARRY:.*]]: tensor<8x8xf32>):
  // CHECK-NEXT:   %[[DOT:.*]] = stablehlo.dot_general %[[CARRY]], %[[IN]], contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT:   stablehlo.return %[[DOT]], %[[CARRY]] : tensor<8xf32>, tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<16x8xf32>, tensor<8x8xf32>) -> (tensor<16x8xf32>, tensor<8x8xf32>)
  // CHECK-NEXT: return %[[SCAN]]#0, %[[SCAN]]#1 : tensor<16x8xf32>, tensor<8x8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = false
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8x8xf32>):
    %1 = stablehlo.dot_general %carry0, %input0, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8xf32>) -> tensor<8xf32>
    stablehlo.return %1, %carry0 : tensor<8xf32>, tensor<8x8xf32>
  } : (tensor<16x8xf32>, tensor<8x8xf32>) -> (tensor<16x8xf32>, tensor<8x8xf32>)
  return %0#0, %0#1 : tensor<16x8xf32>, tensor<8x8xf32>
}
