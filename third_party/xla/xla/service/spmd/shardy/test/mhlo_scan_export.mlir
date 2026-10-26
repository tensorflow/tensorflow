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

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Lowers a 1D forward associative scan sharded along the scan dimension ("x") into 3 phases.
// CHECK-LABEL: func @scan_dim_sharded_forward_1d
// CHECK-SAME: (%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>},
// CHECK-SAME:  %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x, []>})
// CHECK-SAME: -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>},
// CHECK-SAME:     tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x, []>})
func.func @scan_dim_sharded_forward_1d(
    %arg0: tensor<12xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>},
    %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x, []>})
    -> (tensor<12xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>},
        tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x, []>}) {
  // Phase 1: Local reduction per shard.
  // CHECK-NEXT: %[[INIT_SLICE:.*]] = stablehlo.slice %arg0 [0:1] {{.*}}: (tensor<4xf32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[LOCAL_INIT:.*]] = stablehlo.reshape %[[INIT_SLICE]] {{.*}}: (tensor<1xf32>) -> tensor<f32>
  // CHECK-NEXT: %[[REST_SLICE:.*]] = stablehlo.slice %arg0 [1:4] {{.*}}: (tensor<4xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:2 = mhlo.scan(%[[REST_SLICE]]) inits (%[[LOCAL_INIT]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 3 : i64
  // CHECK:      } : (tensor<3xf32>, tensor<f32>) -> (tensor<3xf32>, tensor<f32>)
  // CHECK-NEXT: %[[LOCAL_TOTAL:.*]] = stablehlo.reshape %[[LOCAL_SCAN]]#1 {{.*}}: (tensor<f32>) -> tensor<1xf32>
  //
  // Phase 2: Cross-shard prefix scan.
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL]]) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 3 : i64
  // CHECK:      } : (tensor<3xf32>, tensor<f32>) -> (tensor<3xf32>, tensor<f32>)
  // CHECK-NEXT: %[[INIT_1D:.*]] = stablehlo.reshape %arg1 {{.*}}: (tensor<f32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[PREF_SLICE:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#0 [0:2] {{.*}}: (tensor<3xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[ALL_INC:.*]] = stablehlo.concatenate %[[INIT_1D]], %[[PREF_SLICE]], dim = 0 {{.*}}: (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SHARDED_INC:.*]] = stablehlo.dynamic_slice %[[ALL_INC]], %[[OFFSET_0]], sizes = [1] : (tensor<3xf32>, tensor<i32>) -> tensor<1xf32>
  //
  // Phase 3: Final per-shard scan.
  // CHECK-NEXT: %[[INC_0D:.*]] = stablehlo.reshape %[[SHARDED_INC]] {{.*}}: (tensor<1xf32>) -> tensor<f32>
  // CHECK-NEXT: %[[FINAL_SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%[[INC_0D]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 4 : i64
  // CHECK:      } : (tensor<4xf32>, tensor<f32>) -> (tensor<4xf32>, tensor<f32>)
  // CHECK-NEXT: return %[[FINAL_SCAN]]#0, %[[PREFIX_SCAN]]#1 : tensor<4xf32>, tensor<f32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      scan_dim_size = 12 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}]>, <@mesh_x, []>]>
  } {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<f32>
    stablehlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<12xf32>, tensor<f32>) -> (tensor<12xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<12xf32>, tensor<f32>
}

// -----

sdy.mesh @mesh_xy = <["x"=3, "y"=2]>

// Lowers a 2D scan sharded along both the scan dimension ("x") and non-scan dimension ("y").
// CHECK-LABEL: func @scan_dim_and_non_scan_dim_sharded_2d
// CHECK-SAME: (%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:  %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
// CHECK-SAME: -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
func.func @scan_dim_and_non_scan_dim_sharded_2d(
    %arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
    -> (tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // Phase 1: Local reduction per shard.
  // CHECK-NEXT: %[[INIT_SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:1] : (tensor<4x4xf32>) -> tensor<4x1xf32>
  // CHECK-NEXT: %[[LOCAL_INIT:.*]] = stablehlo.reshape %[[INIT_SLICE]] {{.*}}: (tensor<4x1xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[REST_SLICE:.*]] = stablehlo.slice %arg0 [0:4, 1:4] : (tensor<4x4xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:2 = mhlo.scan(%[[REST_SLICE]]) inits (%[[LOCAL_INIT]]) dimension=1
  // CHECK-NEXT: ^bb0(%[[IN1:.*]]: tensor<4xf32>, %[[CARRY1:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[IN1]], %[[CARRY1]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD1]], %[[ADD1]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<4x3xf32>, tensor<4xf32>) -> (tensor<4x3xf32>, tensor<4xf32>)
  // CHECK-NEXT: %[[LOCAL_TOTAL:.*]] = stablehlo.reshape %[[LOCAL_SCAN]]#1 {{.*}}: (tensor<4xf32>) -> tensor<4x1xf32>
  //
  // Phase 2: Cross-shard prefix scan.
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL]]) <{all_gather_dim = 1 : i64, {{.*}}}> : (tensor<4x1xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=1
  // CHECK-NEXT: ^bb0(%[[IN2:.*]]: tensor<4xf32>, %[[CARRY2:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add %[[IN2]], %[[CARRY2]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD2]], %[[ADD2]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<4x3xf32>, tensor<4xf32>) -> (tensor<4x3xf32>, tensor<4xf32>)
  // CHECK-NEXT: %[[INIT_2D:.*]] = stablehlo.reshape %arg1 {{.*}}: (tensor<4xf32>) -> tensor<4x1xf32>
  // CHECK-NEXT: %[[PREF_SLICE:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#0 [0:4, 0:2] : (tensor<4x3xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %[[ALL_INC:.*]] = stablehlo.concatenate %[[INIT_2D]], %[[PREF_SLICE]], dim = 1 {{.*}}: (tensor<4x1xf32>, tensor<4x2xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 0, 1, 1, 2, 2]> : tensor<6xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<6xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET_1:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[SHARDED_INC:.*]] = stablehlo.dynamic_slice %[[ALL_INC]], %[[OFFSET_0]], %[[OFFSET_1]], sizes = [4, 1] : (tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x1xf32>
  //
  // Phase 3: Final per-shard scan.
  // CHECK-NEXT: %[[INC_1D:.*]] = stablehlo.reshape %[[SHARDED_INC]] {{.*}}: (tensor<4x1xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[FINAL_SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%[[INC_1D]]) dimension=1
  // CHECK-NEXT: ^bb0(%[[IN3:.*]]: tensor<4xf32>, %[[CARRY3:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[ADD3:.*]] = stablehlo.add %[[IN3]], %[[CARRY3]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD3]], %[[ADD3]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<4x4xf32>, tensor<4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %[[FINAL_SCAN]]#0, %[[PREFIX_SCAN]]#1 : tensor<4x4xf32>, tensor<4xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=1 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"y"}, {"x"}]>, <@mesh_xy, [{"y"}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<8x12xf32>, tensor<8xf32>) -> (tensor<8x12xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8x12xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Lowers a reverse associative scan sharded along the scan dimension ("x").
// CHECK-LABEL: func @scan_dim_sharded_reverse
// CHECK-SAME: (%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func @scan_dim_sharded_reverse(
    %arg0: tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // Phase 1: Slice tail [3:4] as local init and [0:3] as rest.
  // CHECK-NEXT: %[[INIT_SLICE:.*]] = stablehlo.slice %arg0 [3:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[LOCAL_INIT:.*]] = stablehlo.reshape %[[INIT_SLICE]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[REST_SLICE:.*]] = stablehlo.slice %arg0 [0:3, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:2 = mhlo.scan(%[[REST_SLICE]]) inits (%[[LOCAL_INIT]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   is_reverse = true
  // CHECK:      } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[LOCAL_TOTAL:.*]] = stablehlo.reshape %[[LOCAL_SCAN]]#1 {{.*}}: (tensor<8xf32>) -> tensor<1x8xf32>
  //
  // Phase 2: All-gather shard totals, run reverse prefix scan, and prepend slice [1:3] before global init.
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL]]) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   is_reverse = true
  // CHECK:      } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[INIT_2D:.*]] = stablehlo.reshape %arg1 {{.*}}: (tensor<8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[PREF_SLICE:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#0 [1:3, 0:8] {{.*}}: (tensor<3x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ALL_INC:.*]] = stablehlo.concatenate %[[PREF_SLICE]], %[[INIT_2D]], dim = 0 {{.*}}: (tensor<2x8xf32>, tensor<1x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SHARDED_INC:.*]] = stablehlo.dynamic_slice %[[ALL_INC]], %[[OFFSET_0]], %[[OFFSET_1]], sizes = [1, 8] : (tensor<3x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
  //
  // Phase 3: Final reverse per-shard scan with incoming carry.
  // CHECK-NEXT: %[[INC_1D:.*]] = stablehlo.reshape %[[SHARDED_INC]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[FINAL_SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%[[INC_1D]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   is_reverse = true
  // CHECK:      } : (tensor<4x8xf32>, tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: return %[[FINAL_SCAN]]#0, %[[PREFIX_SCAN]]#1 : tensor<4x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      is_reverse = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<12x8xf32>, tensor<8xf32>) -> (tensor<12x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<12x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Lowers a variadic scan with multiple inputs and carries sharded along the scan dimension ("x").
// CHECK-LABEL: func @scan_dim_sharded_variadic
// CHECK-SAME: (%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg2: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>},
// CHECK-SAME:  %arg3: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>},
// CHECK-SAME:     tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func @scan_dim_sharded_variadic(
    %arg0: tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg2: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>},
    %arg3: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>},
        tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // Phase 1: Local reduction per shard.
  // CHECK-NEXT: %[[INIT_SLICE0:.*]] = stablehlo.slice %arg0 [0:1, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[LOCAL_INIT0:.*]] = stablehlo.reshape %[[INIT_SLICE0]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[REST_SLICE0:.*]] = stablehlo.slice %arg0 [1:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[INIT_SLICE1:.*]] = stablehlo.slice %arg1 [0:1, 0:8] {{.*}}: (tensor<4x8xi32>) -> tensor<1x8xi32>
  // CHECK-NEXT: %[[LOCAL_INIT1:.*]] = stablehlo.reshape %[[INIT_SLICE1]] {{.*}}: (tensor<1x8xi32>) -> tensor<8xi32>
  // CHECK-NEXT: %[[REST_SLICE1:.*]] = stablehlo.slice %arg1 [1:4, 0:8] {{.*}}: (tensor<4x8xi32>) -> tensor<3x8xi32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:4 = mhlo.scan(%[[REST_SLICE0]], %[[REST_SLICE1]]) inits (%[[LOCAL_INIT0]], %[[LOCAL_INIT1]]) dimension=0
  // CHECK:      } : (tensor<3x8xf32>, tensor<3x8xi32>, tensor<8xf32>, tensor<8xi32>) -> (tensor<3x8xf32>, tensor<3x8xi32>, tensor<8xf32>, tensor<8xi32>)
  // CHECK-NEXT: %[[LOCAL_TOTAL0:.*]] = stablehlo.reshape %[[LOCAL_SCAN]]#2 {{.*}}: (tensor<8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[LOCAL_TOTAL1:.*]] = stablehlo.reshape %[[LOCAL_SCAN]]#3 {{.*}}: (tensor<8xi32>) -> tensor<1x8xi32>
  //
  // Phase 2: Cross-shard prefix scan.
  // CHECK-NEXT: %[[GATHERED0:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL0]]) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[GATHERED1:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL1]]) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1x8xi32>) -> tensor<3x8xi32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:4 = mhlo.scan(%[[GATHERED0]], %[[GATHERED1]]) inits (%arg2, %arg3) dimension=0
  // CHECK:      } : (tensor<3x8xf32>, tensor<3x8xi32>, tensor<8xf32>, tensor<8xi32>) -> (tensor<3x8xf32>, tensor<3x8xi32>, tensor<8xf32>, tensor<8xi32>)
  // CHECK-NEXT: %[[INIT0_2D:.*]] = stablehlo.reshape %arg2 {{.*}}: (tensor<8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[PREF_SLICE0:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#0 [0:2, 0:8] {{.*}}: (tensor<3x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ALL_INC0:.*]] = stablehlo.concatenate %[[INIT0_2D]], %[[PREF_SLICE0]], dim = 0 {{.*}}: (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS0:.*]] = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE0:.*]] = stablehlo.dynamic_slice %[[OFFSETS0]], %[[PID0]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET0_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE0]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET0_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SHARDED_INC0:.*]] = stablehlo.dynamic_slice %[[ALL_INC0]], %[[OFFSET0_0]], %[[OFFSET0_1]], sizes = [1, 8] : (tensor<3x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[INIT1_2D:.*]] = stablehlo.reshape %arg3 {{.*}}: (tensor<8xi32>) -> tensor<1x8xi32>
  // CHECK-NEXT: %[[PREF_SLICE1:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#1 [0:2, 0:8] {{.*}}: (tensor<3x8xi32>) -> tensor<2x8xi32>
  // CHECK-NEXT: %[[ALL_INC1:.*]] = stablehlo.concatenate %[[INIT1_2D]], %[[PREF_SLICE1]], dim = 0 {{.*}}: (tensor<1x8xi32>, tensor<2x8xi32>) -> tensor<3x8xi32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS1:.*]] = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE1:.*]] = stablehlo.dynamic_slice %[[OFFSETS1]], %[[PID1]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET1_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET1_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SHARDED_INC1:.*]] = stablehlo.dynamic_slice %[[ALL_INC1]], %[[OFFSET1_0]], %[[OFFSET1_1]], sizes = [1, 8] : (tensor<3x8xi32>, tensor<i32>, tensor<i32>) -> tensor<1x8xi32>
  //
  // Phase 3: Final per-shard scan.
  // CHECK-NEXT: %[[INC0_1D:.*]] = stablehlo.reshape %[[SHARDED_INC0]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[INC1_1D:.*]] = stablehlo.reshape %[[SHARDED_INC1]] {{.*}}: (tensor<1x8xi32>) -> tensor<8xi32>
  // CHECK-NEXT: %[[FINAL_SCAN:.*]]:4 = mhlo.scan(%arg0, %arg1) inits (%[[INC0_1D]], %[[INC1_1D]]) dimension=0
  // CHECK:      } : (tensor<4x8xf32>, tensor<4x8xi32>, tensor<8xf32>, tensor<8xi32>) -> (tensor<4x8xf32>, tensor<4x8xi32>, tensor<8xf32>, tensor<8xi32>)
  // CHECK-NEXT: return %[[FINAL_SCAN]]#0, %[[FINAL_SCAN]]#1, %[[PREFIX_SCAN]]#2, %[[PREFIX_SCAN]]#3
  %0:4 = mhlo.scan (%arg0, %arg1) inits (%arg2, %arg3) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %input1: tensor<8xi32>, %carry0: tensor<8xf32>, %carry1: tensor<8xi32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    %2 = stablehlo.add %input1, %carry1 : tensor<8xi32>
    stablehlo.return %1, %2, %1, %2 : tensor<8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xi32>
  } : (tensor<12x8xf32>, tensor<12x8xi32>, tensor<8xf32>, tensor<8xi32>) -> (tensor<12x8xf32>, tensor<12x8xi32>, tensor<8xf32>, tensor<8xi32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<12x8xf32>, tensor<12x8xi32>, tensor<8xf32>, tensor<8xi32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Falls back to replicating the scan dimension ("x") when local scan size is 1.
// CHECK-LABEL: func @scan_dim_sharded_local_size_one_fallback
// CHECK-SAME: (%arg0: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func @scan_dim_sharded_local_size_one_fallback(
    %arg0: tensor<3x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<3x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=0
  // CHECK:      } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SLICED:.*]] = stablehlo.dynamic_slice %[[SCAN]]#0, %[[OFFSET_0]], %[[OFFSET_1]], sizes = [1, 8] : (tensor<3x8xf32>, tensor<i32>, tensor<i32>) -> tensor<1x8xf32>
  // CHECK-NEXT: return %[[SLICED]], %[[SCAN]]#1 : tensor<1x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<3x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Falls back to replicating the scan dimension ("x") when terminator output != carry.
// CHECK-LABEL: func @scan_dim_sharded_output_not_equal_carry_fallback
// CHECK-SAME: (%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func @scan_dim_sharded_output_not_equal_carry_fallback(
    %arg0: tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<4x8xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=0
  // CHECK:      } : (tensor<12x8xf32>, tensor<8xf32>) -> (tensor<12x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 4, 8]> : tensor<3xi32>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<3xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[SLICED:.*]] = stablehlo.dynamic_slice %[[SCAN]]#0, %[[OFFSET_0]], %[[OFFSET_1]], sizes = [4, 8] : (tensor<12x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SLICED]], %[[SCAN]]#1 : tensor<4x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    %2 = stablehlo.multiply %1, %input0 : tensor<8xf32>
    stablehlo.return %2, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<12x8xf32>, tensor<8xf32>) -> (tensor<12x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<12x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Pads and partitions mhlo.scan along an indivisible non-scan dimension ("x",
// size 10 on 3 shards).
// CHECK-LABEL: func private @scan_non_scan_dim_sharded_indivisible
// CHECK-SAME: (%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}, {"x"}]>},
// CHECK-SAME:  %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>})
// CHECK-SAME: -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}, {"x"}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>})
func.func private @scan_non_scan_dim_sharded_indivisible(
    %arg0: tensor<16x10xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}, {"x"}]>},
    %arg1: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>})
    -> (tensor<16x10xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}, {"x"}]>},
        tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}]>}) {
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%arg0) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: tensor<4xf32>, %[[CARRY:.*]]: tensor<4xf32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[IN]], %[[CARRY]] : tensor<4xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]], %[[ADD]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } : (tensor<16x4xf32>, tensor<4xf32>) -> (tensor<16x4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %[[SCAN]]#0, %[[SCAN]]#1 : tensor<16x4xf32>, tensor<4xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{}, {"x"}]>, <@mesh_x, [{"x"}]>]>
  } {
  ^bb0(%input0: tensor<10xf32>, %carry0: tensor<10xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<10xf32>
    stablehlo.return %1, %1 : tensor<10xf32>, tensor<10xf32>
  } : (tensor<16x10xf32>, tensor<10xf32>) -> (tensor<16x10xf32>, tensor<10xf32>)
  return %0#0, %0#1 : tensor<16x10xf32>, tensor<10xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Pads and lowers a forward scan along an indivisible scan dimension ("x",
// size 10 on 3 shards -> padded to 12) using outer masking.
// CHECK-LABEL: func private @scan_dim_sharded_indivisible_forward
// CHECK-SAME: (%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func private @scan_dim_sharded_indivisible_forward(
    %arg0: tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // Pad input and compute 1D global scan indices (iota + shard offset).
  // CHECK:      stablehlo.select {{.*}}, %arg0, {{.*}} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK:      %[[BASE_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<4xi32>
  // CHECK:      %[[GLOBAL_IDX:.*]] = stablehlo.add %[[BASE_IOTA]], {{.*}} : tensor<4xi32>
  //
  // Phase 1: Unmasked local reduction of length 3, then outer carry selection
  // on the last shard (V = 2 -> validOutIdx = 0).
  // CHECK-NEXT: %[[LAST_IDX_SLICE:.*]] = stablehlo.slice %[[GLOBAL_IDX]] [3:4] {{.*}}: (tensor<4xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[LAST_IDX:.*]] = stablehlo.reshape %[[LAST_IDX_SLICE]] {{.*}}: (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[LIMIT:.*]] = stablehlo.constant dense<10> : tensor<i32>
  // CHECK-NEXT: %[[IS_LAST_SHARD:.*]] = stablehlo.compare GE, %[[LAST_IDX]], %[[LIMIT]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[INIT_SLICE:.*]] = stablehlo.slice {{.*}} [0:1, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[LOCAL_INIT:.*]] = stablehlo.reshape %[[INIT_SLICE]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[REST_SLICE:.*]] = stablehlo.slice {{.*}} [1:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:2 = mhlo.scan(%[[REST_SLICE]]) inits (%[[LOCAL_INIT]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 3 : i64
  // CHECK-NEXT: ^bb0(%[[IN1:.*]]: tensor<8xf32>, %[[CARRY1:.*]]: tensor<8xf32>):
  // CHECK-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[IN1]], %[[CARRY1]] : tensor<8xf32>
  // CHECK-NEXT:   stablehlo.return %[[MUL1]], %[[MUL1]] : tensor<8xf32>, tensor<8xf32>
  // CHECK-NEXT: } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[VALID_OUT_SLICE:.*]] = stablehlo.slice %[[LOCAL_SCAN]]#0 [0:1, 0:8] {{.*}}: (tensor<3x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[PARTIAL_CARRY:.*]] = stablehlo.reshape %[[VALID_OUT_SLICE]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SHARD_CARRY:.*]] = stablehlo.select %[[IS_LAST_SHARD]], %[[PARTIAL_CARRY]], %[[LOCAL_SCAN]]#1 {{.*}}: tensor<i1>, tensor<8xf32>
  // CHECK-NEXT: %[[LOCAL_TOTAL:.*]] = stablehlo.reshape %[[SHARD_CARRY]] {{.*}}: (tensor<8xf32>) -> tensor<1x8xf32>
  //
  // Phase 2: Cross-shard prefix scan on 3 shard totals.
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL]]) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<1x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:2 = mhlo.scan(%[[GATHERED]]) inits (%arg1) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 3 : i64
  // CHECK:      } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  //
  // Phase 3: Unmasked final per-shard scan of length 4.
  // CHECK:      %[[FINAL_SCAN:.*]]:2 = mhlo.scan({{.*}}) inits ({{.*}}) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   scan_dim_size = 4 : i64
  // CHECK-NEXT: ^bb0(%[[IN3:.*]]: tensor<8xf32>, %[[CARRY3:.*]]: tensor<8xf32>):
  // CHECK-NEXT:   %[[MUL3:.*]] = stablehlo.multiply %[[IN3]], %[[CARRY3]] : tensor<8xf32>
  // CHECK-NEXT:   stablehlo.return %[[MUL3]], %[[MUL3]] : tensor<8xf32>, tensor<8xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>, tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[OUT_SLICE:.*]] = stablehlo.slice %[[FINAL_SCAN]]#0 [0:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[OUT_SLICE]], %[[PREFIX_SCAN]]#1 : tensor<4x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      scan_dim_size = 10 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.multiply %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<10x8xf32>, tensor<8xf32>) -> (tensor<10x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<10x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=3]>

// Pads and lowers a reverse scan along an indivisible scan dimension ("x",
// size 10 on 3 shards -> padded to 12) using outer shifting and carry selection.
// CHECK-LABEL: func private @scan_dim_sharded_indivisible_reverse
// CHECK-SAME: (%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func private @scan_dim_sharded_indivisible_reverse(
    %arg0: tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // Phase 1: Shift last shard's valid slice [0:2] to [2:4], run unmasked
  // reverse local scan, and select validOutIdx = 2 on the last shard.
  // CHECK:      %[[IS_LAST_SHARD1:.*]] = stablehlo.compare GE, {{.*}}, {{.*}} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[VALID_SLICE1:.*]] = stablehlo.slice {{.*}} [0:2, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ZERO1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[SHIFTED1:.*]] = stablehlo.pad %[[VALID_SLICE1]], %[[ZERO1]], low = [2, 0], high = [0, 0], interior = [0, 0] {{.*}}: (tensor<2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[LOCAL_IN1:.*]] = stablehlo.select %[[IS_LAST_SHARD1]], %[[SHIFTED1]], {{.*}} : tensor<i1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[INIT_SLICE1:.*]] = stablehlo.slice %[[LOCAL_IN1]] [3:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[LOCAL_INIT1:.*]] = stablehlo.reshape %[[INIT_SLICE1]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[REST_SLICE1:.*]] = stablehlo.slice %[[LOCAL_IN1]] [0:3, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:2 = mhlo.scan(%[[REST_SLICE1]]) inits (%[[LOCAL_INIT1]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   is_reverse = true
  // CHECK-NEXT: ^bb0(%[[IN1:.*]]: tensor<8xf32>, %[[CARRY1:.*]]: tensor<8xf32>):
  // CHECK-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[IN1]], %[[CARRY1]] : tensor<8xf32>
  // CHECK-NEXT:   stablehlo.return %[[MUL1]], %[[MUL1]] : tensor<8xf32>, tensor<8xf32>
  // CHECK-NEXT: } : (tensor<3x8xf32>, tensor<8xf32>) -> (tensor<3x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[VALID_OUT_SLICE1:.*]] = stablehlo.slice %[[LOCAL_SCAN]]#0 [2:3, 0:8] {{.*}}: (tensor<3x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[PARTIAL_CARRY1:.*]] = stablehlo.reshape %[[VALID_OUT_SLICE1]] {{.*}}: (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SHARD_CARRY1:.*]] = stablehlo.select %[[IS_LAST_SHARD1]], %[[PARTIAL_CARRY1]], %[[LOCAL_SCAN]]#1 {{.*}}: tensor<i1>, tensor<8xf32>
  //
  // Phase 3: Shift last shard's valid slice [0:2] to [2:4], run unmasked
  // reverse final scan, and shift [2:4] back to [0:2].
  // CHECK:      %[[IS_LAST_SHARD3:.*]] = stablehlo.compare GE, {{.*}}, {{.*}} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[VALID_SLICE3:.*]] = stablehlo.slice {{.*}} [0:2, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ZERO3_IN:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[SHIFTED3:.*]] = stablehlo.pad %[[VALID_SLICE3]], %[[ZERO3_IN]], low = [2, 0], high = [0, 0], interior = [0, 0] {{.*}}: (tensor<2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[LOCAL_IN3:.*]] = stablehlo.select %[[IS_LAST_SHARD3]], %[[SHIFTED3]], {{.*}} : tensor<i1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[FINAL_INIT3:.*]] = stablehlo.reshape {{.*}} : (tensor<1x8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[FINAL_SCAN:.*]]:2 = mhlo.scan(%[[LOCAL_IN3]]) inits (%[[FINAL_INIT3]]) dimension=0
  // CHECK-SAME:   is_associative = true
  // CHECK-SAME:   is_reverse = true
  // CHECK-NEXT: ^bb0(%[[IN3:.*]]: tensor<8xf32>, %[[CARRY3:.*]]: tensor<8xf32>):
  // CHECK-NEXT:   %[[MUL3:.*]] = stablehlo.multiply %[[IN3]], %[[CARRY3]] : tensor<8xf32>
  // CHECK-NEXT:   stablehlo.return %[[MUL3]], %[[MUL3]] : tensor<8xf32>, tensor<8xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>, tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[VALID_OUT3:.*]] = stablehlo.slice %[[FINAL_SCAN]]#0 [2:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ZERO3_OUT:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[UNSHIFTED3:.*]] = stablehlo.pad %[[VALID_OUT3]], %[[ZERO3_OUT]], low = [0, 0], high = [2, 0], interior = [0, 0] {{.*}}: (tensor<2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[FINAL_OUT3:.*]] = stablehlo.select %[[IS_LAST_SHARD3]], %[[UNSHIFTED3]], %[[FINAL_SCAN]]#0 {{.*}}: tensor<i1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[OUT_SLICE:.*]] = stablehlo.slice %[[FINAL_OUT3]] [0:4, 0:8] {{.*}}: (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[OUT_SLICE]], {{.*}} : tensor<4x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      is_reverse = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.multiply %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<10x8xf32>, tensor<8xf32>) -> (tensor<10x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<10x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_x = <["x"=4]>

// Falls back to replicating the scan dimension ("x") when a trailing shard has
// 0 valid elements (size 5 on 4 shards).
// CHECK-LABEL: func private @scan_dim_sharded_indivisible_empty_shard_fallback
// CHECK-SAME: (%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
// CHECK-SAME: -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
// CHECK-SAME:     tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
func.func private @scan_dim_sharded_indivisible_empty_shard_fallback(
    %arg0: tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>})
    -> (tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{"x"}, {}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x, [{}]>}) {
  // CHECK-NEXT: %[[GATHERED:.*]] = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, {{.*}}}> : (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[SLICED_IN:.*]] = stablehlo.slice %[[GATHERED]] [0:5, 0:8] {{.*}}: (tensor<8x8xf32>) -> tensor<5x8xf32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = mhlo.scan(%[[SLICED_IN]]) inits (%arg1) dimension=0
  // CHECK:      } : (tensor<5x8xf32>, tensor<8xf32>) -> (tensor<5x8xf32>, tensor<8xf32>)
  // CHECK:      %[[SLICED_OUT:.*]] = stablehlo.dynamic_slice {{.*}}, sizes = [2, 8] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<2x8xf32>
  // CHECK-NEXT: return %[[SLICED_OUT]], %[[SCAN]]#1 : tensor<2x8xf32>, tensor<8xf32>
  %0:2 = mhlo.scan (%arg0) inits (%arg1) dimension=0 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x, [{"x"}, {}]>, <@mesh_x, [{}]>]>
  } {
  ^bb0(%input0: tensor<8xf32>, %carry0: tensor<8xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<8xf32>
    stablehlo.return %1, %1 : tensor<8xf32>, tensor<8xf32>
  } : (tensor<5x8xf32>, tensor<8xf32>) -> (tensor<5x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<5x8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh_xy = <["x"=3, "y"=2]>

// Pads and lowers a 2D mhlo.scan with two inputs when both the non-scan
// dimension (dimension=0, size 7 on 2 shards -> padded to 8) and the scan
// dimension (dimension=1, size 7 on 3 shards -> padded to 9, V = 1) are
// indivisible.
// CHECK-LABEL: func private @scan_dim_sharded_indivisible_multi_input_2d
// CHECK-SAME: (%arg0: tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:  %arg1: tensor<4x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:  %arg2: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>},
// CHECK-SAME:  %arg3: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
// CHECK-SAME: -> (tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:     tensor<4x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
// CHECK-SAME:     tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>},
// CHECK-SAME:     tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
func.func private @scan_dim_sharded_indivisible_multi_input_2d(
    %arg0: tensor<7x7xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
    %arg1: tensor<7x7xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
    %arg2: tensor<7xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>},
    %arg3: tensor<7xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>})
    -> (tensor<7x7xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
        tensor<7x7xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}, {"x"}]>},
        tensor<7xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>},
        tensor<7xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // Pad inputs and inits and compute 1D global scan indices along "x".
  // CHECK:      stablehlo.select {{.*}}, %arg0, {{.*}} : tensor<4x3xi1>, tensor<4x3xf32>
  // CHECK:      stablehlo.select {{.*}}, %arg2, {{.*}} : tensor<4xi1>, tensor<4xf32>
  // CHECK:      stablehlo.select {{.*}}, %arg1, {{.*}} : tensor<4x3xi1>, tensor<4x3xi32>
  // CHECK:      stablehlo.select {{.*}}, %arg3, {{.*}} : tensor<4xi1>, tensor<4xi32>
  // CHECK:      %[[BASE_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<3xi32>
  // CHECK:      %[[GLOBAL_IDX:.*]] = stablehlo.add %[[BASE_IOTA]], {{.*}} : tensor<3xi32>
  //
  // Phase 1: Unmasked local reduction of length 2 along dimension=1,
  // then outer carry selection on the last shard (V = 1 -> selects localInit).
  // CHECK-NEXT: %[[LAST_IDX_SLICE:.*]] = stablehlo.slice %[[GLOBAL_IDX]] [2:3] {{.*}}: (tensor<3xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[LAST_IDX:.*]] = stablehlo.reshape %[[LAST_IDX_SLICE]] {{.*}}: (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT: %[[IS_LAST_SHARD:.*]] = stablehlo.compare GE, %[[LAST_IDX]], %[[LIMIT]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[INIT_SLICE0:.*]] = stablehlo.slice {{.*}} [0:4, 0:1] {{.*}}: (tensor<4x3xf32>) -> tensor<4x1xf32>
  // CHECK-NEXT: %[[LOCAL_INIT0:.*]] = stablehlo.reshape %[[INIT_SLICE0]] {{.*}}: (tensor<4x1xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[REST_SLICE0:.*]] = stablehlo.slice {{.*}} [0:4, 1:3] {{.*}}: (tensor<4x3xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %[[INIT_SLICE1:.*]] = stablehlo.slice {{.*}} [0:4, 0:1] {{.*}}: (tensor<4x3xi32>) -> tensor<4x1xi32>
  // CHECK-NEXT: %[[LOCAL_INIT1:.*]] = stablehlo.reshape %[[INIT_SLICE1]] {{.*}}: (tensor<4x1xi32>) -> tensor<4xi32>
  // CHECK-NEXT: %[[REST_SLICE1:.*]] = stablehlo.slice {{.*}} [0:4, 1:3] {{.*}}: (tensor<4x3xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: %[[LOCAL_SCAN:.*]]:4 = mhlo.scan(%[[REST_SLICE0]], %[[REST_SLICE1]]) inits (%[[LOCAL_INIT0]], %[[LOCAL_INIT1]]) dimension=1
  // CHECK-SAME:   is_associative = true
  // CHECK-NEXT: ^bb0(%[[IN1_0:.*]]: tensor<4xf32>, %[[IN1_1:.*]]: tensor<4xi32>, %[[CARRY1_0:.*]]: tensor<4xf32>, %[[CARRY1_1:.*]]: tensor<4xi32>):
  // CHECK-NEXT:   %[[ADD1_0:.*]] = stablehlo.add %[[IN1_0]], %[[CARRY1_0]] : tensor<4xf32>
  // CHECK-NEXT:   %[[ADD1_1:.*]] = stablehlo.add %[[IN1_1]], %[[CARRY1_1]] : tensor<4xi32>
  // CHECK-NEXT:   stablehlo.return %[[ADD1_0]], %[[ADD1_1]], %[[ADD1_0]], %[[ADD1_1]] : tensor<4xf32>, tensor<4xi32>, tensor<4xf32>, tensor<4xi32>
  // CHECK-NEXT: } : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<4xf32>, tensor<4xi32>) -> (tensor<4x2xf32>, tensor<4x2xi32>, tensor<4xf32>, tensor<4xi32>)
  // CHECK-NEXT: %[[SHARD_CARRY0:.*]] = stablehlo.select %[[IS_LAST_SHARD]], %[[LOCAL_INIT0]], %[[LOCAL_SCAN]]#2 {{.*}}: tensor<i1>, tensor<4xf32>
  // CHECK-NEXT: %[[LOCAL_TOTAL0:.*]] = stablehlo.reshape %[[SHARD_CARRY0]] {{.*}}: (tensor<4xf32>) -> tensor<4x1xf32>
  // CHECK-NEXT: %[[SHARD_CARRY1:.*]] = stablehlo.select %[[IS_LAST_SHARD]], %[[LOCAL_INIT1]], %[[LOCAL_SCAN]]#3 {{.*}}: tensor<i1>, tensor<4xi32>
  // CHECK-NEXT: %[[LOCAL_TOTAL1:.*]] = stablehlo.reshape %[[SHARD_CARRY1]] {{.*}}: (tensor<4xi32>) -> tensor<4x1xi32>
  //
  // Phase 2: Cross-shard prefix scan on 3 shard totals along dimension=1.
  // CHECK-NEXT: %[[GATHERED0:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL0]]) <{all_gather_dim = 1 : i64, {{.*}}}> : (tensor<4x1xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[GATHERED1:.*]] = "stablehlo.all_gather"(%[[LOCAL_TOTAL1]]) <{all_gather_dim = 1 : i64, {{.*}}}> : (tensor<4x1xi32>) -> tensor<4x3xi32>
  // CHECK-NEXT: %[[PREFIX_SCAN:.*]]:4 = mhlo.scan(%[[GATHERED0]], %[[GATHERED1]]) inits ({{.*}}, {{.*}}) dimension=1
  // CHECK-SAME:   is_associative = true
  // CHECK:      } : (tensor<4x3xf32>, tensor<4x3xi32>, tensor<4xf32>, tensor<4xi32>) -> (tensor<4x3xf32>, tensor<4x3xi32>, tensor<4xf32>, tensor<4xi32>)
  //
  // Phase 3: Unmasked final per-shard scan of length 3 along dimension=1.
  // CHECK:      %[[FINAL_SCAN:.*]]:4 = mhlo.scan({{.*}}, {{.*}}) inits ({{.*}}, {{.*}}) dimension=1
  // CHECK-SAME:   is_associative = true
  // CHECK-NEXT: ^bb0(%[[IN3_0:.*]]: tensor<4xf32>, %[[IN3_1:.*]]: tensor<4xi32>, %[[CARRY3_0:.*]]: tensor<4xf32>, %[[CARRY3_1:.*]]: tensor<4xi32>):
  // CHECK-NEXT:   %[[ADD3_0:.*]] = stablehlo.add %[[IN3_0]], %[[CARRY3_0]] : tensor<4xf32>
  // CHECK-NEXT:   %[[ADD3_1:.*]] = stablehlo.add %[[IN3_1]], %[[CARRY3_1]] : tensor<4xi32>
  // CHECK-NEXT:   stablehlo.return %[[ADD3_0]], %[[ADD3_1]], %[[ADD3_0]], %[[ADD3_1]] : tensor<4xf32>, tensor<4xi32>, tensor<4xf32>, tensor<4xi32>
  // CHECK-NEXT: } : (tensor<4x3xf32>, tensor<4x3xi32>, tensor<4xf32>, tensor<4xi32>) -> (tensor<4x3xf32>, tensor<4x3xi32>, tensor<4xf32>, tensor<4xi32>)
  // CHECK-NEXT: %[[OUT_SLICE0:.*]] = stablehlo.slice %[[FINAL_SCAN]]#0 [0:4, 0:3] {{.*}}: (tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %[[OUT_SLICE1:.*]] = stablehlo.slice %[[FINAL_SCAN]]#1 [0:4, 0:3] {{.*}}: (tensor<4x3xi32>) -> tensor<4x3xi32>
  // CHECK-NEXT: %[[CARRY_SLICE0:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#2 [0:4] {{.*}}: (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[CARRY_SLICE1:.*]] = stablehlo.slice %[[PREFIX_SCAN]]#3 [0:4] {{.*}}: (tensor<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[OUT_SLICE0]], %[[OUT_SLICE1]], %[[CARRY_SLICE0]], %[[CARRY_SLICE1]] : tensor<4x3xf32>, tensor<4x3xi32>, tensor<4xf32>, tensor<4xi32>
  %0:4 = mhlo.scan (%arg0, %arg1) inits (%arg2, %arg3) dimension=1 attributes {
      is_associative = true,
      sdy.sharding = #sdy.sharding_per_value<[
          <@mesh_xy, [{"y"}, {"x"}]>, <@mesh_xy, [{"y"}, {"x"}]>,
          <@mesh_xy, [{"y"}]>, <@mesh_xy, [{"y"}]>]>
  } {
  ^bb0(%in0: tensor<7xf32>, %in1: tensor<7xi32>, %carry0: tensor<7xf32>, %carry1: tensor<7xi32>):
    %add0 = stablehlo.add %in0, %carry0 : tensor<7xf32>
    %add1 = stablehlo.add %in1, %carry1 : tensor<7xi32>
    stablehlo.return %add0, %add1, %add0, %add1 : tensor<7xf32>, tensor<7xi32>, tensor<7xf32>, tensor<7xi32>
  } : (tensor<7x7xf32>, tensor<7x7xi32>, tensor<7xf32>, tensor<7xi32>)
      -> (tensor<7x7xf32>, tensor<7x7xi32>, tensor<7xf32>, tensor<7xi32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<7x7xf32>, tensor<7x7xi32>, tensor<7xf32>, tensor<7xi32>
}
