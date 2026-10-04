// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// nv_tensor_ir.reduce preserves the rank, so the reduced dimensions are
// contracted to 1 and a reshape recovers the stablehlo result shape.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_add
// CHECK-SAME: (%[[IN:.*]]: tensor<16x32x64xf32>) -> tensor<16x64xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <add>> : tensor<16x32x64xf32> -> tensor<16x1x64xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<16x1x64xf32> -> tensor<16x64xf32>
// CHECK-NEXT: results %[[RES]] : tensor<16x64xf32>
func.func @reduce_add(%arg0: tensor<16x32x64xf32>) -> tensor<16x64xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<16x32x64xf32>, tensor<f32>) -> tensor<16x64xf32>
  return %0 : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_mul
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <mul>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_mul(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// The init value has to be -inf, the identity of the max reduction.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_max
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <max>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_max(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_min
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <min>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_min(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reduce_i32
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xsi32>) -> tensor<4xsi32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <add>> : tensor<4x8xsi32> -> tensor<4x1xsi32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xsi32> -> tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @reduce_i32(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// The dimensions are narrowed to an i32 dense array and are not sorted.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_multiple_dimensions
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8x16xf32>) -> tensor<8xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [2, 0], reduction_mode = <add>> : tensor<4x8x16xf32> -> tensor<1x8x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<1x8x1xf32> -> tensor<8xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8xf32>
func.func @reduce_multiple_dimensions(%arg0: tensor<4x8x16xf32>) -> tensor<8xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [2, 0] : (tensor<4x8x16xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// A rank-0 graph result is declared as `[1]`, which is exactly the shape the
// reduce already produces, so no reshape survives. See `PromoteRank0Tensors`.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_to_scalar
// CHECK-SAME: (%[[IN:.*]]: tensor<8xf32>) -> tensor<1xf32>
// CHECK: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [0], reduction_mode = <add>> : tensor<8xf32> -> tensor<1xf32>
// CHECK-NEXT: results %[[REDUCE]] : tensor<1xf32>
func.func @reduce_to_scalar(%arg0: tensor<8xf32>) -> tensor<f32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Both reductions are matched, i.e. the elementwise patterns do not rewrite the
// stablehlo.add inside the first body before the second reduce is visited.
// CHECK-LABEL: nv_tensor_ir.graph @two_reductions
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK: %[[REDUCE0:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <add>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES0:.*]] = reshape %[[REDUCE0]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE1:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <max>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES1:.*]] = reshape %[[REDUCE1]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES0]], %[[RES1]] : tensor<4xf32>, tensor<4xf32>
func.func @two_reductions(%arg0: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.reduce(%arg0 init: %neg_inf) applies stablehlo.maximum across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// nv_tensor_ir.reduce_ud. Reductions whose combiner is not one of the four
// built-in modes carry an explicit identity and a region holding the combiner.
//
// The region operates on *signless* scalars even when the reduction itself is
// over a signed or unsigned element type: the op verifier drops the signedness
// of the identity before comparing it against the block argument types. The
// signedness is therefore carried by the identity attribute and the operand
// type only, and the arith operation inside the region is the one that picks
// the signed or unsigned interpretation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: nv_tensor_ir.graph @reduce_and
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xsi32>) -> tensor<4xsi32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [-1 : si32]> (%[[PREV:.*]]: i32, %[[CURR:.*]]: i32) {
// CHECK-NEXT:   %[[AND:.*]] = arith.andi %[[PREV]], %[[CURR]] : i32
// CHECK-NEXT:   nv_tensor_ir.yield %[[AND]] : i32
// CHECK-NEXT: } : tensor<4x8xsi32> -> tensor<4x1xsi32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xsi32> -> tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @reduce_and(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
  %init = stablehlo.constant dense<-1> : tensor<i32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
   reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
    %1 = stablehlo.and %lhs, %rhs : tensor<i32>
    stablehlo.return %1 : tensor<i32>
  }
  return %0 : tensor<4xi32>
}

// -----

// The identity keeps the unsigned element type, the region arguments do not.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_or_unsigned
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xui32>) -> tensor<4xui32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [0 : ui32]> (%[[PREV:.*]]: i32, %[[CURR:.*]]: i32) {
// CHECK-NEXT:   %[[OR:.*]] = arith.ori %[[PREV]], %[[CURR]] : i32
// CHECK-NEXT:   nv_tensor_ir.yield %[[OR]] : i32
// CHECK-NEXT: } : tensor<4x8xui32> -> tensor<4x1xui32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xui32> -> tensor<4xui32>
// CHECK-NEXT: results %[[RES]] : tensor<4xui32>
func.func @reduce_or_unsigned(%arg0: tensor<4x8xui32>) -> tensor<4xui32> {
  %init = stablehlo.constant dense<0> : tensor<ui32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xui32>, tensor<ui32>) -> tensor<4xui32>
   reducer(%lhs: tensor<ui32>, %rhs: tensor<ui32>) {
    %1 = stablehlo.or %lhs, %rhs : tensor<ui32>
    stablehlo.return %1 : tensor<ui32>
  }
  return %0 : tensor<4xui32>
}

// -----

// A multi-operation body is translated operation by operation, including the
// i1 intermediate produced by the comparison.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_compare_select
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [0.000000e+00 : f32]> (%[[PREV:.*]]: f32, %[[CURR:.*]]: f32) {
// CHECK-NEXT:   %[[CMP:.*]] = arith.cmpf olt, %[[PREV]], %[[CURR]] : f32
// CHECK-NEXT:   %[[SEL:.*]] = arith.select %[[CMP]], %[[PREV]], %[[CURR]] : f32
// CHECK-NEXT:   nv_tensor_ir.yield %[[SEL]] : f32
// CHECK-NEXT: } : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_compare_select(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    %1 = stablehlo.compare LT, %lhs, %rhs : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.select %1, %lhs, %rhs : tensor<i1>, tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

// -----

// Constants inside the body become scalar arith.constant, and clamp is
// decomposed the same way as its elementwise counterpart.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_clamp
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [0.000000e+00 : f32]> (%[[PREV:.*]]: f32, %[[CURR:.*]]: f32) {
// CHECK-NEXT:   %[[LO:.*]] = arith.constant -1.000000e+06 : f32
// CHECK-NEXT:   %[[HI:.*]] = arith.constant 1.000000e+06 : f32
// CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[PREV]], %[[CURR]] : f32
// CHECK-NEXT:   %[[LOWER:.*]] = arith.maximumf %[[LO]], %[[SUM]] : f32
// CHECK-NEXT:   %[[CLAMPED:.*]] = arith.minimumf %[[HI]], %[[LOWER]] : f32
// CHECK-NEXT:   nv_tensor_ir.yield %[[CLAMPED]] : f32
// CHECK-NEXT: } : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_clamp(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    %lo = stablehlo.constant dense<-1.000000e+06> : tensor<f32>
    %hi = stablehlo.constant dense<1.000000e+06> : tensor<f32>
    %1 = stablehlo.add %lhs, %rhs : tensor<f32>
    %2 = stablehlo.clamp %lo, %1, %hi : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

// -----

// nv_tensor_ir.reduce has no init operand and implicitly uses the identity of
// its reduction mode, so an add reduction seeded with 1.0 cannot use it and
// falls back to reduce_ud, whose identity is explicit.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_non_identity_init
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [1.000000e+00 : f32]> (%[[PREV:.*]]: f32, %[[CURR:.*]]: f32) {
// CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[PREV]], %[[CURR]] : f32
// CHECK-NEXT:   nv_tensor_ir.yield %[[SUM]] : f32
// CHECK-NEXT: } : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_non_identity_init(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// add(%prev, %prev) ignores the incoming element, so it is not the built-in
// add reduction even though the combiner and the identity both look like one.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_repeated_argument
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce_ud(%[[IN]]) <dimensions = [1], identity = [0.000000e+00 : f32]> (%[[PREV:.*]]: f32, %[[CURR:.*]]: f32) {
// CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[PREV]], %[[PREV]] : f32
// CHECK-NEXT:   nv_tensor_ir.yield %[[SUM]] : f32
// CHECK-NEXT: } : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_repeated_argument(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    %1 = stablehlo.add %lhs, %lhs : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

// -----

// A flipped built-in combiner still uses the fast path: add and multiply are
// commutative, so the operand order carries no information. This is a
// deliberate divergence from the string-building converter, which required
// add(%prev, %curr) exactly and fell back to reduce_ud otherwise.
// CHECK-LABEL: nv_tensor_ir.graph @reduce_add_flipped
// CHECK-SAME: (%[[IN:.*]]: tensor<4x8xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[REDUCE:.*]] = reduce(%[[IN]]) <dimensions = [1], reduction_mode = <add>> : tensor<4x8xf32> -> tensor<4x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[REDUCE]] : tensor<4x1xf32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @reduce_add_flipped(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    %1 = stablehlo.add %rhs, %lhs : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

