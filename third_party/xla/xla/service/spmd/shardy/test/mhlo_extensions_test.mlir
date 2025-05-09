// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -sdy-insert-explicit-reshards -verify-diagnostics 2>&1 | FileCheck %s
// TODO(enver): Seperate into a sharding rule and explicit reshard tests.

sdy.mesh @mesh_abcd = <["a"=2, "b"=2, "c"=2, "d"=2]>

// CHECK-LABEL: func @copy
func.func @copy(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=8}>
  %0 = mhlo.copy %arg0 : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_non_contracting
func.func @ragged_dot_mode_non_contracting(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<4x16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]>},
    %arg2: tensor<16x4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}]>}) -> (tensor<16x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>}) {
  // CHECK: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_abcd, [{"a"}, {}, {"c"}]> : tensor<16x32x64xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_abcd, [{}, {"a"}, {"c"}, {"d"}]> : tensor<4x16x64x8xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_abcd, [{"a"}, {}]> : tensor<16x4xi32>

  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%[[RESHARD0]], %[[RESHARD1]], %[[RESHARD2]]) <{
  // CHECK: }>
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {}, {"d"}]>]>
  // CHECK-SAME: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [m, i, l, k], [i, m])->([i, j, k]) {i=16, j=32, k=8, l=64, m=4} reduction={l} need_replication={j, m}>

  // CHECK: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"c"} %[[RAGGED_DOT]] out_sharding=<@mesh_abcd, [{"a"}, {}, {"d"}]> : tensor<16x32x8xf32>
  // CHECK: %[[RESHARD3:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_abcd, [{"a"}, {"b"}, {"c"}]> : tensor<16x32x8xf32>
  // CHECK: return %[[RESHARD3]] : tensor<16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>]>}
    : (tensor<16x32x64xf32>, tensor<4x16x64x8xf32>, tensor<16x4xi32>) -> tensor<16x32x8xf32>
  return %0 : tensor<16x32x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_contracting
func.func @ragged_dot_mode_contracting(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg2: tensor<16x4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}]>}) -> (tensor<4x16x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]>}) {
  // CHECK: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_abcd, [{"a"}, {"b"}, {}]> : tensor<16x32x64xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_abcd, [{"a"}, {}, {"d"}]> : tensor<16x64x8xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_abcd, [{"a"}, {}]> : tensor<16x4xi32>

  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%[[RESHARD0]], %[[RESHARD1]], %[[RESHARD2]]) <{
  // CHECK: }>
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{}, {"a"}, {"b"}, {"d"}]>]>
  // CHECK-SAME: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k], [i, m])->([m, i, j, k]) {i=16, j=32, k=8, l=64, m=4} need_replication={l, m}>

  // CHECK: %[[RESHARD3:.*]] = sdy.reshard %[[RAGGED_DOT]] <@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]> : tensor<4x16x32x8xf32>
  // CHECK: return %[[RESHARD3]] : tensor<4x16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    lhs_ragged_dimensions = [2]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"c"}, {"d"}]>]>}
    : (tensor<16x32x64xf32>, tensor<16x64x8xf32>, tensor<16x4xi32>) -> tensor<4x16x32x8xf32>
  return %0 : tensor<4x16x32x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_batch
func.func @ragged_dot_mode_batch(
    %arg0: tensor<16x32x64xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"c"}]>},
    %arg1: tensor<16x64x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"c"}, {"d"}]>},
    %arg2: tensor<4xi32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}]>}) -> (tensor<16x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh_abcd, [{"a"}, {"b"}, {"d"}]>}) {
  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{
  // CHECK: }>
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"d"}]>]>
  // CHECK-SAME: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k], [m])->([i, j, k]) {i=16, j=32, k=8, l=64, m=1} reduction={l}>
  // CHECK: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"c"} %[[RAGGED_DOT]] out_sharding=<@mesh_abcd, [{"a"}, {"b"}, {"d"}]> : tensor<16x32x8xf32>
  // CHECK: return %[[ALL_REDUCE]] : tensor<16x32x8xf32>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    lhs_ragged_dimensions = [0]>}>
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abcd, [{"a"}, {"b"}, {"d"}]>]>}
    : (tensor<16x32x64xf32>, tensor<16x64x8xf32>, tensor<4xi32>) -> tensor<16x32x8xf32>
  return %0 : tensor<16x32x8xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_non_contracting_multiple_dims
func.func @ragged_dot_mode_non_contracting_multiple_dims(%arg0: tensor<41x21x11x42x22x12x43xf32>, %arg1: tensor<41x11x31x42x32x12x43x7xf32>, %arg2: tensor<11x12x21x7xi32>) -> tensor<11x12x21x22x31x32xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([o, k, i, p, l, j, q], [o, i, m, p, n, j, q, r], [i, j, k, r])->([i, j, k, l, m, n]) {i=11, j=12, k=21, l=22, m=31, n=32, o=41, p=42, q=43, r=7} reduction={o, p, q} need_replication={l, r}>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [2, 5], rhs_batching_dimensions = [1, 5],
    lhs_contracting_dimensions = [0, 3, 6], rhs_contracting_dimensions = [0, 3, 6]>,
    lhs_ragged_dimensions = [4], rhs_group_dimensions = [7]>}>
    : (tensor<41x21x11x42x22x12x43xf32>, tensor<41x11x31x42x32x12x43x7xf32>, tensor<11x12x21x7xi32>) -> tensor<11x12x21x22x31x32xf32>
  return %0 : tensor<11x12x21x22x31x32xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_contracting_multiple_dims
func.func @ragged_dot_mode_contracting_multiple_dims(%arg0: tensor<41x21x11x42x22x12x43xf32>, %arg1: tensor<41x11x31x42x32x12x43xf32>, %arg2: tensor<11x12x41x7xi32>) -> tensor<7x11x12x21x22x31x32xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([o, k, i, p, l, j, q], [o, i, m, p, n, j, q], [i, j, o, r])->([r, i, j, k, l, m, n]) {i=11, j=12, k=21, l=22, m=31, n=32, o=41, p=42, q=43, r=7} reduction={o, q} need_replication={p, r}>}
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [2, 5], rhs_batching_dimensions = [1, 5],
    lhs_contracting_dimensions = [0, 3, 6], rhs_contracting_dimensions = [0, 3, 6]>,
    lhs_ragged_dimensions = [3]>}>
    : (tensor<41x21x11x42x22x12x43xf32>, tensor<41x11x31x42x32x12x43xf32>, tensor<11x12x41x7xi32>) -> tensor<7x11x12x21x22x31x32xf32>
  return %0 : tensor<7x11x12x21x22x31x32xf32>
}

// CHECK-LABEL: func @ragged_dot_mode_batch_multiple_dims
func.func @ragged_dot_mode_batch_multiple_dims(%arg0: tensor<41x21x11x42x22x12x43xf32>, %arg1: tensor<41x11x31x42x32x12x43xf32>, %arg2: tensor<11x7xi32>) -> tensor<11x12x21x22x31x32xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([o, k, i, p, l, j, q], [o, i, m, p, n, j, q], [r, s])->([i, j, k, l, m, n]) {i=11, j=12, k=21, l=22, m=31, n=32, o=41, p=42, q=43, r=1, s=1} reduction={o, p, q}>
  %0 = "mhlo.ragged_dot"(%arg0, %arg1, %arg2) <{ragged_dot_dimension_numbers =
    #mhlo.ragged_dot<dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [2, 5], rhs_batching_dimensions = [1, 5],
    lhs_contracting_dimensions = [0, 3, 6], rhs_contracting_dimensions = [0, 3, 6]>,
    lhs_ragged_dimensions = [5]>}>
    : (tensor<41x21x11x42x22x12x43xf32>, tensor<41x11x31x42x32x12x43xf32>, tensor<11x7xi32>) -> tensor<11x12x21x22x31x32xf32>
  return %0 : tensor<11x12x21x22x31x32xf32>
}
