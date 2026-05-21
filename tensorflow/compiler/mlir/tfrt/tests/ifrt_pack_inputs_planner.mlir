// RUN: tf-tfrt-opt %s -ifrt-pack-inputs-planner="size-threshold-bytes=4096" | FileCheck %s
// RUN: tf-tfrt-opt %s -ifrt-pack-inputs-planner="size-threshold-bytes=16" \
// RUN:   | FileCheck %s --check-prefix=TIGHT

// Mix of large + small + variable operands.
// %big (40000 bytes) is over the 4 KiB threshold                → -1
// %sm_a (24 bytes), %sm_b (24 bytes) under threshold            →  0
// %var (4 bytes) is in variable_arg_indices (already on device) → -1
// CHECK-LABEL: func.func @mix
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [-1, 0, 0, -1]
// CHECK-SAME: variable_arg_indices = [3 : i32]
func.func @mix(%big: tensor<100x100xf32>,
               %sm_a: tensor<3x2xf32>,
               %sm_b: tensor<3x2xf32>,
               %var: tensor<f32>) -> tensor<3x2xf32> {
  %0 = "tf.IfrtCall"(%big, %sm_a, %sm_b, %var) {
    operandSegmentSizes = array<i32: 4, 0>,
    program_id = 1 : i64,
    variable_arg_indices = [3 : i32]
  } : (tensor<100x100xf32>, tensor<3x2xf32>, tensor<3x2xf32>, tensor<f32>)
      -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// All small and non-variable — all go to group 0.
// CHECK-LABEL: func.func @all_small
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [0, 0, 0]
func.func @all_small(%a: tensor<2xf32>,
                     %b: tensor<2xi32>,
                     %c: tensor<i64>) -> tensor<2xf32> {
  %0 = "tf.IfrtCall"(%a, %b, %c) {
    operandSegmentSizes = array<i32: 3, 0>,
    program_id = 2 : i64,
    variable_arg_indices = []
  } : (tensor<2xf32>, tensor<2xi32>, tensor<i64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// Filter 1 violations: dynamic shape, sub-byte bitwidth, unranked → -1.
// Only %ok (rank=1, static, byte-aligned) packs.
// CHECK-LABEL: func.func @ineligible
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [-1, -1, -1, 0]
func.func @ineligible(%dyn: tensor<?xf32>,
                      %sub_byte: tensor<8xi1>,
                      %unranked: tensor<*xf32>,
                      %ok: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.IfrtCall"(%dyn, %sub_byte, %unranked, %ok) {
    operandSegmentSizes = array<i32: 4, 0>,
    program_id = 3 : i64,
    variable_arg_indices = []
  } : (tensor<?xf32>, tensor<8xi1>, tensor<*xf32>, tensor<2xf32>)
      -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// Tight threshold (16 bytes): only the smallest two pack.
// %a (8 bytes), %b (8 bytes)                  → 0
// %c (24 bytes) > 16 bytes                    → -1
// TIGHT-LABEL: func.func @threshold
// TIGHT: "tf.IfrtCall"
// TIGHT-SAME: ifrt_pack_group_ids = [0, 0, -1]
func.func @threshold(%a: tensor<2xf32>,
                     %b: tensor<i64>,
                     %c: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = "tf.IfrtCall"(%a, %b, %c) {
    operandSegmentSizes = array<i32: 3, 0>,
    program_id = 4 : i64,
    variable_arg_indices = []
  } : (tensor<2xf32>, tensor<i64>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Filter 4 — layout sensitivity. %m1 and %m2 are small enough to pack on
// size, but %m1 feeds tf.MatMul in the atom func, so it's excluded.
// %m2 only feeds tf.AddV2 (pointwise) → allowed.
// CHECK-LABEL: func.func @layout_fussy
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [-1, 0]
func.func private @atom_with_matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>
    attributes {tfrt_ifrt_serving.program_id = 7 : i64} {
  %0 = "tf.MatMul"(%arg0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "tf.AddV2"(%0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
func.func @layout_fussy(%m1: tensor<4x4xf32>, %m2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "tf.IfrtCall"(%m1, %m2) {
    operandSegmentSizes = array<i32: 2, 0>,
    program_id = 7 : i64,
    variable_arg_indices = []
  } : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Bounded dynamism: arg flows through tf.SetStaticDimensionBounds whose
// static_shape operand is a tf.Const carrying [8, 4]. Element type is
// quint8 (1 byte). Upper-bound byte size = 8*4*1 = 32 bytes, well under
// the 4 KiB threshold → packs (group 0).
//
// This test simulates the state AFTER PropagateStaticShapesPass has run:
// 1. tf.SetStaticDimensionBounds is gone (replaced by its input).
// 2. IfrtCall has the bound in its static_shapes segment.
// 3. Callee has tf._static_shape_arg_idx.
// CHECK-LABEL: func.func @bounded_quint8
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [0]
func.func private @callee_with_bound(%arg0: tensor<?x?x!tf_type.quint8>, %arg1: tensor<2xi32>) -> tensor<?x?x!tf_type.quint8>
    attributes {tfrt_ifrt_serving.program_id = 8 : i64} {
  %0 = "tf.Identity"(%arg0) {tf._static_shape_arg_idx = 1 : i32} : (tensor<?x?x!tf_type.quint8>) -> tensor<?x?x!tf_type.quint8>
  return %0 : tensor<?x?x!tf_type.quint8>
}
func.func @bounded_quint8(%dyn: tensor<?x?x!tf_type.quint8>) -> tensor<?x?x!tf_type.quint8> {
  %bound = "tf.Const"() {value = dense<[8, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.IfrtCall"(%dyn, %bound) {
    operandSegmentSizes = array<i32: 1, 1>,
    program_id = 8 : i64,
    variable_arg_indices = []
  } : (tensor<?x?x!tf_type.quint8>, tensor<2xi32>) -> tensor<?x?x!tf_type.quint8>
  return %1 : tensor<?x?x!tf_type.quint8>
}

// Multiple IfrtCalls in one module — each gets its own annotation.
// CHECK-LABEL: func.func @two_calls
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [0, 0]
// CHECK: "tf.IfrtCall"
// CHECK-SAME: ifrt_pack_group_ids = [0]
func.func @two_calls(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.IfrtCall"(%a, %b) {
    operandSegmentSizes = array<i32: 2, 0>,
    program_id = 5 : i64,
    variable_arg_indices = []
  } : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %1 = "tf.IfrtCall"(%0) {
    operandSegmentSizes = array<i32: 1, 0>,
    program_id = 6 : i64,
    variable_arg_indices = []
  } : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}
