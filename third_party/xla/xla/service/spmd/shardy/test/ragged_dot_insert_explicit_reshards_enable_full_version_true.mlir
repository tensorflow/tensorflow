// RUN: sdy_opt %s -allow-unregistered-dialect  -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>
sdy.mesh @other_mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh_abcd = <["a"=2, "b"=2, "c"=2, "d"=2]>

// CHECK-LABEL: func @ragged_dot_mode_non_contracting
func.func @ragged_dot_mode_non_contracting(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<4x16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]>},
    %arg2: tensor<16x4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}]>}) -> tensor<16x32x8xf32> {
  // CHECK: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_abcd, [{"a"}, {}, {"c"}]> : tensor<16x32x64xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_abcd, [{}, {"a"}, {"c"}, {"d"}]> : tensor<4x16x64x8xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_abcd, [{"a"}, {}]> : tensor<16x4xi32>

  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%[[RESHARD0]], %[[RESHARD1]], %[[RESHARD2]]) <{
  // CHECK: }>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {}, {"d"}]>]>

  // CHECK: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"c"} %[[RAGGED_DOT]] out_sharding=<@mesh_abcd, [{"a"}, {}, {"d"}]> : tensor<16x32x8xf32>
  // CHECK: %[[RESHARD3:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_abcd, [{"a"}, {"b"}, {"c"}]> : tensor<16x32x8xf32>
  // CHECK-NEXT: %[[RESHARD4:.*]] = sdy.reshard %[[RESHARD3]] <@mesh_abcd, [{}, {}, {}]> : tensor<16x32x8xf32>
  // CHECK: return %[[RESHARD4]] : tensor<16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [m, i, l, k], [i, m])->([i, j, k]) {i=16, j=32, k=8, l=64, m=4} reduction={l} need_replication={j, m}>,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>]>}
    : (tensor<16x32x64xf32>, tensor<4x16x64x8xf32>, tensor<16x4xi32>) -> tensor<16x32x8xf32>
  return %0 : tensor<16x32x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_contracting
func.func @ragged_dot_mode_contracting(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg2: tensor<16x4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}]>}) -> tensor<4x16x32x8xf32> {
  // CHECK: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_abcd, [{"a"}, {"b"}, {}]> : tensor<16x32x64xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_abcd, [{"a"}, {}, {"d"}]> : tensor<16x64x8xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_abcd, [{"a"}, {}]> : tensor<16x4xi32>

  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%[[RESHARD0]], %[[RESHARD1]], %[[RESHARD2]]) <{
  // CHECK: }>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{}, {"a"}, {"b"}, {"d"}]>]>

  // CHECK: %[[RESHARD3:.*]] = sdy.reshard %[[RAGGED_DOT]] <@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]> : tensor<4x16x32x8xf32>
  // CHECK: %[[RESHARD4:.*]] = sdy.reshard %[[RESHARD3]] <@mesh_abcd, [{}, {}, {}, {}]> : tensor<4x16x32x8xf32>
  // CHECK: return %[[RESHARD4]] : tensor<4x16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    lhs_ragged_dimensions = [2]>}>
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k], [i, m])->([m, i, j, k]) {i=16, j=32, k=8, l=64, m=4} need_replication={l, m}>,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]>]>}
    : (tensor<16x32x64xf32>, tensor<16x64x8xf32>, tensor<16x4xi32>) -> tensor<4x16x32x8xf32>
  return %0 : tensor<4x16x32x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_batch
func.func @ragged_dot_mode_batch(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"c"}, {"d"}]>},
    %arg2: tensor<4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}]>}) -> tensor<16x32x8xf32> {
  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{
  // CHECK: }>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"d"}]>]>
  // CHECK: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"c"} %[[RAGGED_DOT]] out_sharding=<@mesh_abcd, [{"a"}, {"b"}, {"d"}]> : tensor<16x32x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_abcd, [{}, {}, {}]> : tensor<16x32x8xf32>
  // CHECK: return %[[RESHARD]] : tensor<16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    lhs_ragged_dimensions = [0]>}>
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k], [m])->([i, j, k]) {i=16, j=32, k=8, l=64, m=1} reduction={l}>,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"d"}]>]>}
    : (tensor<16x32x64xf32>, tensor<16x64x8xf32>, tensor<4xi32>) -> tensor<16x32x8xf32>
  return %0 : tensor<16x32x8xf32>
}
