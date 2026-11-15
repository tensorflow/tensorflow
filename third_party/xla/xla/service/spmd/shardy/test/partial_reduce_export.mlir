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
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' -sdy-resolve-permutation-factors -sdy-reshard-to-collectives -sdy-pad-for-divisibility -sdy-convert-global-to-local | FileCheck %s
// RUN: sdy_opt %s -sdy-resolve-permutation-factors -sdy-reshard-to-collectives -sdy-pad-for-divisibility -sdy-convert-global-to-local | FileCheck %s

// Tests Shardy global-to-local export of `stablehlo.custom_call @PartialReduce`:
// 1. `@partial_reduce_db_sharded_has_stage_one_comparator`: Stage 1 +
//    all-gather + Stage 2 pad/reshape/reduce with `"comparator"`
//    `stablehlo.select`.
// 2. `@partial_reduce_db_sharded_no_stage_one`: Skips Stage 1 when per-shard
//    `log2_reduction == 0`; all-gathers and folds shards directly.
// 3. `@partial_reduce_uneven_2d_batch_and_db_sharded`: 2D batch + database
//    sharding with high-side divisibility padding, output batch slicing, and
//    resharding to an unannotated (replicated) result.
// 4. `@partial_reduce_default_combiner_single_input`: Single-input (`K = 1`)
//    omitting `to_apply_type` defaults to `"combiner"`.
// 5. `@partial_reduce_tuple_combiner`: Multi-input `"combiner"` returning an
//    HLO-style `stablehlo.tuple` unpacks into Stage 2 `stablehlo.reduce`.
// 6. `@partial_reduce_single_input_tuple_combiner`: Single-input (`K = 1`)
//    `"combiner"` returning `tuple<tensor<f32>>` unpacks cleanly.
// 7. `@partial_reduce_single_fold_pad_only`: `numFolds == 1` pads gathered
//    candidates without emitting `stablehlo.reshape` or `stablehlo.reduce`.
// 8. `@partial_reduce_sub_axis_replicates_reduction_dim`: Sub-axis reduction
//    sharding falls back to replicating `reduction_dim` before single-stage op.
// 9. `@partial_reduce_batch_sharded_only_strips_xla_shape`: Unsharded
//    `reduction_dim` stays single-stage and strips stale `xla_shape`.

sdy.mesh @mesh_d4 = <["d"=4]>
sdy.mesh @mesh_d8 = <["d"=8]>
sdy.mesh @mesh_xy = <["x"=4, "y"=4]>
sdy.mesh @mesh_b2 = <["b"=2]>

func.func private @gt_comparator(%lhs_val: tensor<f32>, %rhs_val: tensor<f32>,
                                 %lhs_idx: tensor<i32>, %rhs_idx: tensor<i32>)
    -> tensor<i1> {
  %0 = stablehlo.compare GT, %lhs_val, %rhs_val : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func.func private @max_reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)
    -> tensor<f32> {
  %0 = stablehlo.maximum %lhs, %rhs : tensor<f32>
  return %0 : tensor<f32>
}

func.func private @pair_max_tuple_combiner(
    %lhs0: tensor<f32>, %lhs1: tensor<f32>,
    %rhs0: tensor<f32>, %rhs1: tensor<f32>)
    -> tuple<tensor<f32>, tensor<f32>> {
  %m0 = stablehlo.maximum %lhs0, %rhs0 : tensor<f32>
  %m1 = stablehlo.maximum %lhs1, %rhs1 : tensor<f32>
  %t = stablehlo.tuple %m0, %m1 : tuple<tensor<f32>, tensor<f32>>
  return %t : tuple<tensor<f32>, tensor<f32>>
}

func.func private @single_max_tuple_combiner(%lhs: tensor<f32>,
                                             %rhs: tensor<f32>)
    -> tuple<tensor<f32>> {
  %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
  %t = stablehlo.tuple %m : tuple<tensor<f32>>
  return %t : tuple<tensor<f32>>
}

// `[4, 10000]` on `"d"=4` (`2500`/shard) -> Stage 1 `[4, 128]` (`log2=5`) ->
// all-gather `[4, 512]` -> pad +256 `[4, 768]` -> fold `[4, 2, 384] -> [4, 384]`.
// CHECK-LABEL: func.func @partial_reduce_db_sharded_has_stage_one_comparator(
// CHECK-SAME:      %arg0: tensor<4x2500xf32>
// CHECK-SAME:      %arg1: tensor<4x2500xi32>
func.func @partial_reduce_db_sharded_has_stage_one_comparator(
    %arg0: tensor<4x10000xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>},
    %arg1: tensor<4x10000xi32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>})
    -> (tensor<4x384xf32>, tensor<4x384xi32>) {
  // CHECK-DAG: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK-DAG: %[[NEG_ONE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // CHECK: %[[STAGE_ONE:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %[[NEG_INF]], %[[NEG_ONE]])
  // CHECK-SAME: backend_config = "{\22log2_reduction\22:5,\22recall_target\22:0.94999999999999996,\22reduction_dim\22:1,\22to_apply_type\22:\22comparator\22,\22top_k\22:10}"
  // CHECK-SAME: called_computations = [@gt_comparator]
  // CHECK-NOT: xla_shape
  // CHECK-SAME: : (tensor<4x2500xf32>, tensor<4x2500xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x128xf32>, tensor<4x128xi32>)
  // CHECK: %[[AG0:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#0)
  // CHECK-SAME: : (tensor<4x128xf32>) -> tensor<4x512xf32>
  // CHECK: %[[AG1:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#1)
  // CHECK-SAME: : (tensor<4x128xi32>) -> tensor<4x512xi32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %[[AG0]], %[[NEG_INF]], low = [0, 0], high = [0, 256], interior = [0, 0] {{.*}}: (tensor<4x512xf32>, tensor<f32>) -> tensor<4x768xf32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %[[AG1]], %[[NEG_ONE]], low = [0, 0], high = [0, 256], interior = [0, 0] {{.*}}: (tensor<4x512xi32>, tensor<i32>) -> tensor<4x768xi32>
  // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[PAD0]] {{.*}}: (tensor<4x768xf32>) -> tensor<4x2x384xf32>
  // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[PAD1]] {{.*}}: (tensor<4x768xi32>) -> tensor<4x2x384xi32>
  // CHECK: %[[RED:.*]]:2 = stablehlo.reduce(%[[RESHAPE0]] init: %[[NEG_INF]]), (%[[RESHAPE1]] init: %[[NEG_ONE]]) across dimensions = [1]
  // CHECK-SAME: : (tensor<4x2x384xf32>, tensor<4x2x384xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x384xf32>, tensor<4x384xi32>)
  // CHECK:   %[[CMP:.*]] = stablehlo.compare GT, %arg2, %arg4
  // CHECK:   %[[SEL0:.*]] = stablehlo.select %[[CMP]], %arg2, %arg4 : tensor<i1>, tensor<f32>
  // CHECK:   %[[SEL1:.*]] = stablehlo.select %[[CMP]], %arg3, %arg5 : tensor<i1>, tensor<i32>
  // CHECK:   stablehlo.return %[[SEL0]], %[[SEL1]] : tensor<f32>, tensor<i32>
  // CHECK: return %[[RED]]#0, %[[RED]]#1 : tensor<4x384xf32>, tensor<4x384xi32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %init_idx = stablehlo.constant dense<-1> : tensor<i32>
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %init_val, %init_idx) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@gt_comparator],
    xla_shape = "(f32[4,384]{1,0}, s32[4,384]{1,0})",
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_d4, [{}, {}]>, <@mesh_d4, [{}, {}]>]>
  } : (tensor<4x10000xf32>, tensor<4x10000xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x384xf32>, tensor<4x384xi32>)
  return %0#0, %0#1 : tensor<4x384xf32>, tensor<4x384xi32>
}

// `[4, 1000]` on `"d"=8` (`125`/shard, `log2=0`) -> skip Stage 1 ->
// all-gather `[4, 1000]` -> pad +24 `[4, 1024]` -> fold `[4, 4, 256] -> [4, 256]`.
// CHECK-LABEL: func.func @partial_reduce_db_sharded_no_stage_one(
// CHECK-SAME:      %arg0: tensor<4x125xf32>
// CHECK-SAME:      %arg1: tensor<4x125xi32>
func.func @partial_reduce_db_sharded_no_stage_one(
    %arg0: tensor<4x1000xf32> {sdy.sharding = #sdy.sharding<@mesh_d8, [{}, {"d"}]>},
    %arg1: tensor<4x1000xi32> {sdy.sharding = #sdy.sharding<@mesh_d8, [{}, {"d"}]>})
    -> (tensor<4x256xf32>, tensor<4x256xi32>) {
  // CHECK-NOT: @PartialReduce
  // CHECK: %[[AG0:.*]] = "stablehlo.all_gather"(%arg0)
  // CHECK-SAME: : (tensor<4x125xf32>) -> tensor<4x1000xf32>
  // CHECK: %[[AG1:.*]] = "stablehlo.all_gather"(%arg1)
  // CHECK-SAME: : (tensor<4x125xi32>) -> tensor<4x1000xi32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %[[AG0]], {{.*}}, low = [0, 0], high = [0, 24], interior = [0, 0] {{.*}}: (tensor<4x1000xf32>, tensor<f32>) -> tensor<4x1024xf32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %[[AG1]], {{.*}}, low = [0, 0], high = [0, 24], interior = [0, 0] {{.*}}: (tensor<4x1000xi32>, tensor<i32>) -> tensor<4x1024xi32>
  // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[PAD0]] {{.*}}: (tensor<4x1024xf32>) -> tensor<4x4x256xf32>
  // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[PAD1]] {{.*}}: (tensor<4x1024xi32>) -> tensor<4x4x256xi32>
  // CHECK: %[[RED:.*]]:2 = stablehlo.reduce(%[[RESHAPE0]] init: {{.*}}), (%[[RESHAPE1]] init: {{.*}}) across dimensions = [1]
  // CHECK-SAME: : (tensor<4x4x256xf32>, tensor<4x4x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x256xf32>, tensor<4x256xi32>)
  // CHECK: return %[[RED]]#0, %[[RED]]#1 : tensor<4x256xf32>, tensor<4x256xi32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %init_idx = stablehlo.constant dense<-1> : tensor<i32>
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %init_val, %init_idx) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 2, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@gt_comparator],
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_d8, [{}, {}]>, <@mesh_d8, [{}, {}]>]>
  } : (tensor<4x1000xf32>, tensor<4x1000xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x256xf32>, tensor<4x256xi32>)
  return %0#0, %0#1 : tensor<4x256xf32>, tensor<4x256xi32>
}

// `[6, 2003]` on `["x"=4, "y"=4]` with omitted (replicated) result sharding ->
// pad to `[8, 2004]` (`[2, 501]`/shard) -> Stage 1 `[2, 128]` -> all-gather
// `"y"` `[2, 512]` -> fold `[2, 256]` -> all-gather `"x"` `[8, 256]` -> slice
// `[6, 256]`.
// CHECK-LABEL: func.func @partial_reduce_uneven_2d_batch_and_db_sharded(
// CHECK-SAME:      %arg0: tensor<6x2003xf32>
// CHECK-SAME:      %arg1: tensor<6x2003xi32>
func.func @partial_reduce_uneven_2d_batch_and_db_sharded(
    %arg0: tensor<6x2003xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {}]>},
    %arg1: tensor<6x2003xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {}]>})
    -> (tensor<6x256xf32>, tensor<6x256xi32>) {
  // CHECK: %[[STAGE_ONE:.*]]:2 = stablehlo.custom_call @PartialReduce
  // CHECK-SAME: : (tensor<2x501xf32>, tensor<2x501xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x128xf32>, tensor<2x128xi32>)
  // CHECK: %[[AG_Y0:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#0)
  // CHECK-SAME: : (tensor<2x128xf32>) -> tensor<2x512xf32>
  // CHECK: %[[AG_Y1:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#1)
  // CHECK-SAME: : (tensor<2x128xi32>) -> tensor<2x512xi32>
  // CHECK: %[[RED:.*]]:2 = stablehlo.reduce({{.*}}) across dimensions = [1]
  // CHECK-SAME: : (tensor<2x2x256xf32>, tensor<2x2x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x256xf32>, tensor<2x256xi32>)
  // CHECK-DAG: %[[SLICE0:.*]] = stablehlo.slice {{.*}} [0:6, 0:256] {{.*}}: (tensor<8x256xf32>) -> tensor<6x256xf32>
  // CHECK-DAG: %[[SLICE1:.*]] = stablehlo.slice {{.*}} [0:6, 0:256] {{.*}}: (tensor<8x256xi32>) -> tensor<6x256xi32>
  // CHECK: return %[[SLICE0]], %[[SLICE1]] : tensor<6x256xf32>, tensor<6x256xi32>
  %sharded0 = sdy.reshard %arg0 <@mesh_xy, [{"x"}, {"y"}]> : tensor<6x2003xf32>
  %sharded1 = sdy.reshard %arg1 <@mesh_xy, [{"x"}, {"y"}]> : tensor<6x2003xi32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %init_idx = stablehlo.constant dense<-1> : tensor<i32>
  %0:2 = stablehlo.custom_call @PartialReduce(%sharded0, %sharded1, %init_val, %init_idx) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 3, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@gt_comparator]
  } : (tensor<6x2003xf32>, tensor<6x2003xi32>, tensor<f32>, tensor<i32>) -> (tensor<6x256xf32>, tensor<6x256xi32>)
  return %0#0, %0#1 : tensor<6x256xf32>, tensor<6x256xi32>
}

// Single-input (`K = 1`) omitting `to_apply_type` defaults to `"combiner"`:
// `[8, 2048]` on `"d"=4` (`512`/shard) -> Stage 1 `[8, 128]` -> all-gather
// `[8, 512]` -> fold `[8, 2, 256] -> [8, 256]`.
// CHECK-LABEL: func.func @partial_reduce_default_combiner_single_input(
// CHECK-SAME:      %arg0: tensor<8x512xf32>
func.func @partial_reduce_default_combiner_single_input(
    %arg0: tensor<8x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>})
    -> tensor<8x256xf32> {
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[STAGE_ONE:.*]] = stablehlo.custom_call @PartialReduce(%arg0, %[[INIT]])
  // CHECK-SAME: backend_config = "{\22log2_reduction\22:2,\22recall_target\22:0.94999999999999996,\22reduction_dim\22:1,\22top_k\22:10}"
  // CHECK-SAME: : (tensor<8x512xf32>, tensor<f32>) -> tensor<8x128xf32>
  // CHECK: %[[AG:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]])
  // CHECK-SAME: : (tensor<8x128xf32>) -> tensor<8x512xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[AG]] {{.*}}: (tensor<8x512xf32>) -> tensor<8x2x256xf32>
  // CHECK: %[[RED:.*]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1]
  // CHECK-SAME: : (tensor<8x2x256xf32>, tensor<f32>) -> tensor<8x256xf32>
  // CHECK: return %[[RED]] : tensor<8x256xf32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.custom_call @PartialReduce(%arg0, %init_val) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 3, \22reduction_dim\22: 1, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@max_reducer],
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_d4, [{}, {}]>]>
  } : (tensor<8x2048xf32>, tensor<f32>) -> tensor<8x256xf32>
  return %0 : tensor<8x256xf32>
}

// Multi-input `"combiner"` returning an HLO-style `stablehlo.tuple` unwraps
// the tuple operands directly into `stablehlo.reduce`'s `stablehlo.return`.
// CHECK-LABEL: func.func @partial_reduce_tuple_combiner(
// CHECK-SAME:      %arg0: tensor<4x512xf32>
// CHECK-SAME:      %arg1: tensor<4x512xf32>
func.func @partial_reduce_tuple_combiner(
    %arg0: tensor<4x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>},
    %arg1: tensor<4x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>})
    -> (tensor<4x256xf32>, tensor<4x256xf32>) {
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[STAGE_ONE:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %[[INIT]], %[[INIT]])
  // CHECK-SAME: backend_config = "{\22log2_reduction\22:2,\22recall_target\22:0.94999999999999996,\22reduction_dim\22:1,\22to_apply_type\22:\22combiner\22,\22top_k\22:10}"
  // CHECK-SAME: : (tensor<4x512xf32>, tensor<4x512xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x128xf32>, tensor<4x128xf32>)
  // CHECK: %[[AG0:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#0)
  // CHECK: %[[AG1:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]]#1)
  // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[AG0]] {{.*}}: (tensor<4x512xf32>) -> tensor<4x2x256xf32>
  // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[AG1]] {{.*}}: (tensor<4x512xf32>) -> tensor<4x2x256xf32>
  // CHECK: %[[RED:.*]]:2 = stablehlo.reduce(%[[RESHAPE0]] init: %[[INIT]]), (%[[RESHAPE1]] init: %[[INIT]]) across dimensions = [1]
  // CHECK-NOT: stablehlo.tuple
  // CHECK:   %[[M0:.*]] = stablehlo.maximum %arg2, %arg4 : tensor<f32>
  // CHECK:   %[[M1:.*]] = stablehlo.maximum %arg3, %arg5 : tensor<f32>
  // CHECK:   stablehlo.return %[[M0]], %[[M1]] : tensor<f32>, tensor<f32>
  // CHECK: return %[[RED]]#0, %[[RED]]#1 : tensor<4x256xf32>, tensor<4x256xf32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %init_val, %init_val) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 3, \22reduction_dim\22: 1, \22to_apply_type\22: \22combiner\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@pair_max_tuple_combiner],
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_d4, [{}, {}]>, <@mesh_d4, [{}, {}]>]>
  } : (tensor<4x2048xf32>, tensor<4x2048xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x256xf32>, tensor<4x256xf32>)
  return %0#0, %0#1 : tensor<4x256xf32>, tensor<4x256xf32>
}

// Single-input (`K = 1`) `"combiner"` returning `tuple<tensor<f32>>` unwraps
// the 1-element tuple into `stablehlo.reduce`.
// CHECK-LABEL: func.func @partial_reduce_single_input_tuple_combiner(
// CHECK-SAME:      %arg0: tensor<8x512xf32>
func.func @partial_reduce_single_input_tuple_combiner(
    %arg0: tensor<8x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d"}]>})
    -> tensor<8x256xf32> {
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[STAGE_ONE:.*]] = stablehlo.custom_call @PartialReduce(%arg0, %[[INIT]])
  // CHECK-SAME: : (tensor<8x512xf32>, tensor<f32>) -> tensor<8x128xf32>
  // CHECK: %[[AG:.*]] = "stablehlo.all_gather"(%[[STAGE_ONE]])
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[AG]] {{.*}}: (tensor<8x512xf32>) -> tensor<8x2x256xf32>
  // CHECK: %[[RED:.*]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1]
  // CHECK-SAME: : (tensor<8x2x256xf32>, tensor<f32>) -> tensor<8x256xf32>
  // CHECK: return %[[RED]] : tensor<8x256xf32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.custom_call @PartialReduce(%arg0, %init_val) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 3, \22reduction_dim\22: 1, \22to_apply_type\22: \22combiner\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@single_max_tuple_combiner]
  } : (tensor<8x2048xf32>, tensor<f32>) -> tensor<8x256xf32>
  return %0 : tensor<8x256xf32>
}

// `[4, 250]` on `"b"=2` (`125`/shard, `log2=0`) -> all-gather `[4, 250]` ->
// `numFolds = ceil(250 / 256) = 1`: high-pad +6 to `[4, 256]` without `reduce`.
// CHECK-LABEL: func.func @partial_reduce_single_fold_pad_only(
// CHECK-SAME:      %arg0: tensor<4x125xf32>
func.func @partial_reduce_single_fold_pad_only(
    %arg0: tensor<4x250xf32> {sdy.sharding = #sdy.sharding<@mesh_b2, [{}, {"b"}]>})
    -> tensor<4x256xf32> {
  // CHECK-NOT: @PartialReduce
  // CHECK-NOT: stablehlo.reduce
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[AG:.*]] = "stablehlo.all_gather"(%arg0)
  // CHECK-SAME: : (tensor<4x125xf32>) -> tensor<4x250xf32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %[[AG]], %[[INIT]], low = [0, 0], high = [0, 6], interior = [0, 0] {{.*}}: (tensor<4x250xf32>, tensor<f32>) -> tensor<4x256xf32>
  // CHECK: return %[[PAD]] : tensor<4x256xf32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.custom_call @PartialReduce(%arg0, %init_val) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 0, \22reduction_dim\22: 1, \22to_apply_type\22: \22combiner\22, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@max_reducer],
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_b2, [{}, {}]>]>
  } : (tensor<4x250xf32>, tensor<f32>) -> tensor<4x256xf32>
  return %0 : tensor<4x256xf32>
}

// Sub-axis sharding `"d":(1)2` on `reduction_dim` falls back to replicating
// `reduction_dim` via `all_gather` before running single-stage `@PartialReduce`.
// CHECK-LABEL: func.func @partial_reduce_sub_axis_replicates_reduction_dim(
// CHECK-SAME:      %arg0: tensor<8x1024xf32>
func.func @partial_reduce_sub_axis_replicates_reduction_dim(
    %arg0: tensor<8x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_d4, [{}, {"d":(1)2}]>})
    -> tensor<8x256xf32> {
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[AG:.*]] = "stablehlo.all_gather"(%arg0)
  // CHECK-SAME: : (tensor<8x1024xf32>) -> tensor<8x2048xf32>
  // CHECK: %[[PR:.*]] = stablehlo.custom_call @PartialReduce(%[[AG]], %[[INIT]])
  // CHECK-NOT: xla_shape
  // CHECK-SAME: : (tensor<8x2048xf32>, tensor<f32>) -> tensor<8x256xf32>
  // CHECK: return %[[PR]] : tensor<8x256xf32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.custom_call @PartialReduce(%arg0, %init_val) {
    api_version = 1 : i32,
    backend_config = "{\22log2_reduction\22: 3, \22reduction_dim\22: 1, \22top_k\22: 10, \22recall_target\22: 0.95}",
    called_computations = [@max_reducer],
    xla_shape = "f32[8,256]{1,0}",
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_d4, [{}, {}]>]>
  } : (tensor<8x2048xf32>, tensor<f32>) -> tensor<8x256xf32>
  return %0 : tensor<8x256xf32>
}

// Batch-only sharding `[{"b"}, {}]` leaves `reduction_dim` unsharded, keeping a
// single-stage `@PartialReduce` and stripping stale global `xla_shape`.
// CHECK-LABEL: func.func @partial_reduce_batch_sharded_only_strips_xla_shape(
// CHECK-SAME:      %arg0: tensor<4x2048xf32>
// CHECK-SAME:      %arg1: tensor<4x2048xi32>
func.func @partial_reduce_batch_sharded_only_strips_xla_shape(
    %arg0: tensor<8x2048xf32> {sdy.sharding = #sdy.sharding<@mesh_b2, [{"b"}, {}]>},
    %arg1: tensor<8x2048xi32> {sdy.sharding = #sdy.sharding<@mesh_b2, [{"b"}, {}]>})
    -> (tensor<8x256xf32> {sdy.sharding = #sdy.sharding<@mesh_b2, [{"b"}, {}]>},
        tensor<8x256xi32> {sdy.sharding = #sdy.sharding<@mesh_b2, [{"b"}, {}]>}) {
  // CHECK-NOT: all_gather
  // CHECK-NOT: stablehlo.reduce
  // CHECK-DAG: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK-DAG: %[[NEG_ONE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // CHECK: %[[PR:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %[[NEG_INF]], %[[NEG_ONE]])
  // CHECK-SAME: called_computations = [@gt_comparator]
  // CHECK-NOT: xla_shape
  // CHECK-SAME: : (tensor<4x2048xf32>, tensor<4x2048xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x256xf32>, tensor<4x256xi32>)
  // CHECK: return %[[PR]]#0, %[[PR]]#1 : tensor<4x256xf32>, tensor<4x256xi32>
  %init_val = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %init_idx = stablehlo.constant dense<-1> : tensor<i32>
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %init_val, %init_idx) {
    api_version = 1 : i32,
    backend_config = "{\22reduction_dim\22: 1}",
    called_computations = [@gt_comparator],
    xla_shape = "(f32[8,256]{1,0}, s32[8,256]{1,0})",
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_b2, [{"b"}, {}]>, <@mesh_b2, [{"b"}, {}]>]>
  } : (tensor<8x2048xf32>, tensor<8x2048xi32>, tensor<f32>, tensor<i32>) -> (tensor<8x256xf32>, tensor<8x256xi32>)
  return %0#0, %0#1 : tensor<8x256xf32>, tensor<8x256xi32>
}
