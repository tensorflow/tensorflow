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
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg2 <@mesh_abcd, [{}]> : tensor<4xi32>
  // CHECK: %[[RAGGED_DOT:.*]] = "mhlo.ragged_dot"(%arg0, %arg1, %[[RESHARD]]) <{
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

// CHECK-LABEL: func @topk_1d
func.func @topk_1d(%arg0: tensor<10xf32>) -> tensor<3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [i]) {i=10} need_replication={i} blocked_propagation={i}>
  %0:2 = mhlo.topk(%arg0, k=3, largest=true) : tensor<10xf32> -> (tensor<3xf32>, tensor<3xi32>)
  return %0#0 : tensor<3xf32>
}

// CHECK-LABEL: func @topk_1d_k_equals_one
func.func @topk_1d_k_equals_one(%arg0: tensor<10xf32>) -> tensor<1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [i]) {i=10} need_replication={i} blocked_propagation={i}>
  %0:2 = mhlo.topk(%arg0, k=1, largest=true) : tensor<10xf32> -> (tensor<1xf32>, tensor<1xi32>)
  return %0#0 : tensor<1xf32>
}

// CHECK-LABEL: func @topk_2d
func.func @topk_2d(%arg0: tensor<16x10xf32>) -> tensor<16x3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=10} need_replication={j} blocked_propagation={j}>
  %0:2 = mhlo.topk(%arg0, k=3, largest=true) : tensor<16x10xf32> -> (tensor<16x3xf32>, tensor<16x3xi32>)
  return %0#0 : tensor<16x3xf32>
}

// CHECK-LABEL: func @topk_2d_k_equals_one
func.func @topk_2d_k_equals_one(%arg0: tensor<16x10xf32>) -> tensor<16x1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=10} need_replication={j} blocked_propagation={j}>
  %0:2 = mhlo.topk(%arg0, k=1, largest=true) : tensor<16x10xf32> -> (tensor<16x1xf32>, tensor<16x1xi32>)
  return %0#0 : tensor<16x1xf32>
}

// Associative scans expose the scan dim as a permutation factor (parallel-prefix
// can shard it at the cost of inter-shard combine traffic).
// CHECK-LABEL: func @scan_1d_associative
func.func @scan_1d_associative(%input: tensor<10xf32>, %init: tensor<f32>)
    -> tensor<10xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [])->([i], []) {i=10} permutation={i}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 attributes {
      is_associative = true
  } {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<f32>
    stablehlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10xf32>, tensor<f32>)
  return %0#0 : tensor<10xf32>
}

// Non-associative scans require sequential evaluation along the scan dim, so
// it must be replicated.
// CHECK-LABEL: func @scan_1d_non_associative
func.func @scan_1d_non_associative(%input: tensor<10xf32>, %init: tensor<f32>)
    -> tensor<10xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [])->([i], []) {i=10} need_replication={i}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 attributes {
      is_associative = false
  } {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<f32>
    stablehlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10xf32>, tensor<f32>)
  return %0#0 : tensor<10xf32>
}

// Missing `is_associative` defaults to non-associative for the rule (safe).
// CHECK-LABEL: func @scan_1d_unspecified_is_associative
func.func @scan_1d_unspecified_is_associative(%input: tensor<10xf32>, %init: tensor<f32>)
    -> tensor<10xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [])->([i], []) {i=10} need_replication={i}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<f32>
    stablehlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10xf32>, tensor<f32>)
  return %0#0 : tensor<10xf32>
}

// 2D shape-preserving scan with the scan dim as the leading (major) dim. The
// non-scan dim is shared across input/output/init/carry as a pass-through
// factor, allowing it to be sharded freely.
// CHECK-LABEL: func @scan_2d_scan_dim_0
func.func @scan_2d_scan_dim_0(%input: tensor<2x3xf32>, %init: tensor<3xf32>)
    -> tensor<2x3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [j])->([i, j], [j]) {i=2, j=3} permutation={i}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 attributes {
      is_associative = true
  } {
  ^bb0(%input0: tensor<3xf32>, %carry0: tensor<3xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<3xf32>
    stablehlo.return %1, %1 : tensor<3xf32>, tensor<3xf32>
  } : (tensor<2x3xf32>, tensor<3xf32>) -> (tensor<2x3xf32>, tensor<3xf32>)
  return %0#0 : tensor<2x3xf32>
}

// 2D shape-preserving scan with the scan dim as the trailing (minor) dim.
// CHECK-LABEL: func @scan_2d_scan_dim_1
func.func @scan_2d_scan_dim_1(%input: tensor<3x2xf32>, %init: tensor<3xf32>)
    -> tensor<3x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i])->([i, j], [i]) {i=3, j=2} permutation={j}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=1 attributes {
      is_associative = true
  } {
  ^bb0(%input0: tensor<3xf32>, %carry0: tensor<3xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<3xf32>
    stablehlo.return %1, %1 : tensor<3xf32>, tensor<3xf32>
  } : (tensor<3x2xf32>, tensor<3xf32>) -> (tensor<3x2xf32>, tensor<3xf32>)
  return %0#0 : tensor<3x2xf32>
}

// 3D shape-preserving scan with the scan dim in the middle. Both surrounding
// dims pass through identically across input/output/init/carry.
// CHECK-LABEL: func @scan_3d_scan_dim_middle
func.func @scan_3d_scan_dim_middle(%input: tensor<4x5x6xf32>, %init: tensor<4x6xf32>)
    -> tensor<4x5x6xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, k])->([i, j, k], [i, k]) {i=4, j=5, k=6} permutation={j}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=1 attributes {
      is_associative = true
  } {
  ^bb0(%input0: tensor<4x6xf32>, %carry0: tensor<4x6xf32>):
    %1 = stablehlo.add %input0, %carry0 : tensor<4x6xf32>
    stablehlo.return %1, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  } : (tensor<4x5x6xf32>, tensor<4x6xf32>) -> (tensor<4x5x6xf32>, tensor<4x6xf32>)
  return %0#0 : tensor<4x5x6xf32>
}

// Multi-input/output shape-preserving scan: every input/output/init/carry
// shares the same dim factors, the scan dim still permutation, the non-scan
// dim still pass-through.
// CHECK-LABEL: func @scan_2d_multi_input
func.func @scan_2d_multi_input(%input1: tensor<2x3xf32>, %input2: tensor<2x3xf32>,
                               %init1: tensor<3xf32>, %init2: tensor<3xf32>)
    -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [j], [j])->([i, j], [i, j], [j], [j]) {i=2, j=3} permutation={i}>
  %0:4 = mhlo.scan (%input1, %input2) inits (%init1, %init2) dimension=0 attributes {
      is_associative = true
  } {
  ^bb0(%in1: tensor<3xf32>, %in2: tensor<3xf32>, %c1: tensor<3xf32>, %c2: tensor<3xf32>):
    %a = stablehlo.add %in1, %c1 : tensor<3xf32>
    %b = stablehlo.add %in2, %c2 : tensor<3xf32>
    stablehlo.return %a, %b, %a, %b : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
  } : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
   -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xf32>
}

// Shape-changing scan (output shape != input shape): falls back to the
// conservative rule that only models the scan dim and leaves every other
// dim with a fresh size-1 factor that nothing can propagate through.
// CHECK-LABEL: func @scan_2d_shape_changing
func.func @scan_2d_shape_changing(%input: tensor<10xf32>, %init: tensor<f32>)
    -> tensor<10x3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [])->([i, j], []) {i=10, j=1} permutation={i}>
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 attributes {
      is_associative = true
  } {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    %1 = stablehlo.broadcast_in_dim %input0, dims = [] : (tensor<f32>) -> tensor<3xf32>
    stablehlo.return %1, %carry0 : tensor<3xf32>, tensor<f32>
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10x3xf32>, tensor<f32>)
  return %0#0 : tensor<10x3xf32>
}
