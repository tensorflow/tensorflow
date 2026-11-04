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
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -sdy-insert-explicit-reshards='enable-full-version=true' -verify-diagnostics | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xpq = <["x"=4, "p"=3, "q"=5]>

// CHECK-LABEL: func @custom_call_compact_wy_helper
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=128, j=128}>
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @custom_call_x64_combine
func.func @custom_call_x64_combine(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) {backend_config = ""} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}

// CHECK-LABEL: func @custom_call_x64_combine_used_by_rng_bit_generator
func.func @custom_call_x64_combine_used_by_rng_bit_generator(%arg0: tensor<2xui32>, %arg1: tensor<2xui32>) -> (tensor<4x1000xui32>, tensor<2xui64>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [i])->([i]) {i=2} need_replication={i} blocked_propagation={i}>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui64>
  %output_state, %output = stablehlo.rng_bit_generator %0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  return %output, %0 : tensor<4x1000xui32>, tensor<2xui64>
}

// CHECK-LABEL: func @custom_call_x64_split_high
func.func @custom_call_x64_split_high(%arg0: tensor<8x2xui64>) -> tensor<8x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = ""} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_x64_split_low
func.func @custom_call_x64_split_low(%arg0: tensor<8x2xui64>) -> tensor<8x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = ""} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_compute_on_done
func.func @custom_call_compute_on_done(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.custom_call @"compute-on-done"(%arg0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_qr_decomposition_block
func.func @custom_call_qr_decomposition_block(%arg0: tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0:2 = stablehlo.custom_call @QrDecompositionBlock(%arg0) : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_lu_real_decomposition_block
func.func @custom_call_lu_real_decomposition_block(%arg0: tensor<4x8x5x3xf32>) -> (tensor<4x8x5x3xf32>, tensor<4x8x3xi32>, tensor<4x8x5xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, j, k, l], [i, j, m], [i, j, k]) {i=4, j=8, k=5, l=3, m=3} need_replication={k, l, m}>
  %0:3 = stablehlo.custom_call @LuRealDecompositionBlock(%arg0) : (tensor<4x8x5x3xf32>) -> (tensor<4x8x5x3xf32>, tensor<4x8x3xi32>, tensor<4x8x5xi32>)
  return %0#0, %0#1, %0#2 : tensor<4x8x5x3xf32>, tensor<4x8x3xi32>, tensor<4x8x5xi32>
}

// CHECK-LABEL: func @custom_call_lu_complex_decomposition_block
func.func @custom_call_lu_complex_decomposition_block(%arg0: tensor<8x5x3xf32>, %arg1: tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x5x3xf32>, tensor<8x3xi32>, tensor<8x5xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k], [i, j, k], [i, l], [i, j]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0:4 = stablehlo.custom_call @LuComplexDecompositionBlock(%arg0, %arg1) : (tensor<8x5x3xf32>, tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x5x3xf32>, tensor<8x3xi32>, tensor<8x5xi32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<8x5x3xf32>, tensor<8x5x3xf32>, tensor<8x3xi32>, tensor<8x5xi32>
}

// CHECK-LABEL: func @custom_call_topk_target_name
func.func @custom_call_topk_target_name(%arg0: tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  %0:2 = stablehlo.custom_call @TopK(%arg0) {
    mhlo.attributes = {
        k = 2 : i64,
        largest = true},
    mhlo.version = 1 : i64}
    : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xi32>
}

// CHECK-LABEL: func @custom_call_partial_reduce
func.func @custom_call_partial_reduce(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce_string_backend_config
func.func @custom_call_partial_reduce_string_backend_config(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 2, \22recall_target\22: 0.950000}",
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_compact_wy_helper_insert_explicit_reshards
func.func @custom_call_compact_wy_helper_insert_explicit_reshards(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=128, j=128}>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<128x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<128x128xf32>
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @custom_call_x64_combine_insert_explicit_reshards
func.func @custom_call_x64_combine_insert_explicit_reshards(%arg0: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> (tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xui32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64Combine(%arg0, %[[RESHARD]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  // CHECK-NEXT: return %[[CUSTOM_CALL]] : tensor<8x2xui64>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}

// CHECK-LABEL: func @custom_call_x64_split_high_insert_explicit_reshards
func.func @custom_call_x64_split_high_insert_explicit_reshards(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_x64_split_low_insert_explicit_reshards
func.func @custom_call_x64_split_low_insert_explicit_reshards(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_qr_decomposition_block_insert_explicit_reshards
func.func @custom_call_qr_decomposition_block_insert_explicit_reshards(%arg0: tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}) -> (tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}, tensor<8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"p"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @QrDecompositionBlock(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>, <@mesh_xpq, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh_xpq, [{"x"}, {"q"}, {"p"}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh_xpq, [{"x"}, {"p"}]> : tensor<8x3xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x5x3xf32>, tensor<8x3xf32>
  %0:2 = stablehlo.custom_call @QrDecompositionBlock(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>, <@mesh_xpq, [{"x"}, {"p"}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce_insert_explicit_reshards
func.func @custom_call_partial_reduce_insert_explicit_reshards(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(2)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {"x":(2)2}]>
  // CHECK-NEXT: %[[PARTIAL_REDUCE:.*]]:2 = stablehlo.custom_call @PartialReduce(%[[RESHARD1]], %arg1, %arg2, %arg3)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce_string_backend_config_insert_explicit_reshards
func.func @custom_call_partial_reduce_string_backend_config_insert_explicit_reshards(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %[[RESHARD1]], %arg2, %arg3)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 2, \22recall_target\22: 0.950000}",
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @unknown_custom_op
func.func @unknown_custom_op(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // expected-warning@+1 {{custom call @unknown_custom_op is unknown to SDY sharding rule registry}}
  %0 = stablehlo.custom_call @unknown_custom_op(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}
