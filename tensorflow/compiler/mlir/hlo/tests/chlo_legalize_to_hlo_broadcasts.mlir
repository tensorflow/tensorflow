// RUN: mlir-hlo-opt -chlo-legalize-to-hlo="legalize-broadcasts=true expand-compositions=false" -cse -canonicalize -split-input-file -verify-diagnostics %s -o - | FileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.
// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.add %arg0, %arg1
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcast
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:  %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-NEXT: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK-DAG:    %[[RESULT_EXTENTS:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-DAG:    %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG:    %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-NEXT:   %[[RESULT:.+]] = mhlo.add %[[ARG0_B]], %[[ARG1_B]]
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK-NEXT:      return %[[FINAL_RESULT]] : tensor<?x?xf32>
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  // CHECK-DAG:  %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:  %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-NEXT: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-DAG:    %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG:    %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT:   %[[RESULT:.+]] = "mhlo.complex"(%[[ARG0_B]], %[[ARG1_B]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FINAL_RESULT]] : tensor<?x?xcomplex<f32>>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // CHECK-DAG: %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK: %[[RESULT_EXTENTS:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-DAG: %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = "mhlo.compare"(%[[ARG0_B]], %[[ARG1_B]]) {comparison_direction = "EQ"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK: return %[[FINAL_RESULT]] : tensor<?x?xi1>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----

// CHECK-LABEL: func @selectv2
func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: "mhlo.select"(%arg0, %arg1, %arg2)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_pred_scalar
func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: "mhlo.select"(%arg0, %arg1, %arg2)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_then
func @selectv2_broadcast_then(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK-NEXT: %[[BROADCAST:.*]] = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x1xi32>) -> tensor<2x8x8xi32>
  // CHECK-NEXT: "mhlo.select"(%arg0, %[[BROADCAST]], %arg2)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_else
func @selectv2_broadcast_else(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  // CHECK-NEXT: %[[BROADCAST:.*]] = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x1xi32>) -> tensor<2x8x8xi32>
  // CHECK-NEXT: "mhlo.select"(%arg0, %arg1, %[[BROADCAST]])
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_pred
func @selectv2_broadcast_pred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK-NEXT: %[[BROADCAST:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<2x8x8xi1>
  // CHECK-NEXT: "mhlo.select"(%[[BROADCAST]], %arg1, %arg2)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_tensor_pred
func @selectv2_broadcast_tensor_pred(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf16>, %arg2: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // CHECK-NEXT: %[[BROADCAST:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK-NEXT: "mhlo.select"(%[[BROADCAST]], %arg1, %arg2)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %0: tensor<2x3xf16>
}

// CHECK-LABEL: func @selectv2_broadcast_all
func @selectv2_broadcast_all(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  // CHECK-DAG: %[[BROADCAST_0:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x1xi1>) -> tensor<8x8x8xi1>
  // CHECK-DAG: %[[BROADCAST_1:.*]] = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x8x1xi32>) -> tensor<8x8x8xi32>
  // CHECK-DAG: %[[BROADCAST_2:.*]] = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  // CHECK: "mhlo.select"(%[[BROADCAST_0]], %[[BROADCAST_1]], %[[BROADCAST_2]])
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  return %0: tensor<8x8x8xi32>
}

// CHECK-LABEL: func @selectv2_dynamic_ranked
func @selectv2_dynamic_ranked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  // CHECK-NEXT: %[[SHAPE0:.*]] = shape.const_shape [1] : tensor<1xindex>
  // CHECK-NEXT: %[[SHAPE2:.*]] = shape.const_shape [2, 8, 8] : tensor<3xindex>
  // CHECK-NEXT: %[[SHAPE1:.*]] = shape.shape_of %arg1 : tensor<2x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[CSTR:.*]] = shape.cstr_broadcastable %[[SHAPE1]], %[[SHAPE0]], %[[SHAPE2]] : tensor<3xindex>, tensor<1xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[ASSUME:.*]] = shape.assuming %[[CSTR]] -> (tensor<2x?x8xi32>) {
  // CHECK-NEXT:   %[[BCST:.*]] = shape.broadcast %[[SHAPE1]], %[[SHAPE2]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
  // CHECK-NEXT:   %[[BCST0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[BCST]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x?x8xi1>
  // CHECK-NEXT:   %[[BCST1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[BCST]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x?x8xi32>, tensor<3xindex>) -> tensor<2x?x8xi32>
  // CHECK-NEXT:   %[[BCST2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg2, %[[BCST]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x8x8xi32>, tensor<3xindex>) -> tensor<2x?x8xi32>
  // CHECK-NEXT:   %[[SELECT:.*]] = "mhlo.select"(%[[BCST0]], %[[BCST1]], %[[BCST2]]) : (tensor<2x?x8xi1>, tensor<2x?x8xi32>, tensor<2x?x8xi32>) -> tensor<2x?x8xi32>
  // CHECK-NEXT:   shape.assuming_yield %[[SELECT]] : tensor<2x?x8xi32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[ASSUME]] : tensor<2x?x8xi32>
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  return %0: tensor<2x?x8xi32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarBroadcastDimensions
func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK: mhlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  // CHECK: mhlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.and %arg0, %arg1
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.atan2 %arg0, %arg1
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // CHECK: "mhlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.divide %arg0, %arg1
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.maximum %arg0, %arg1
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.minimum %arg0, %arg1
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.multiply %arg0, %arg1
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.or %arg0, %arg1
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.power %arg0, %arg1
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.remainder %arg0, %arg1
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_leftWithoutBroadcast
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_left %arg0, %arg1
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_arithmeticWithoutBroadcast
func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_right_arithmetic %arg0, %arg1
  %0 = chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_logicalWithoutBroadcast
func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_right_logical %arg0, %arg1
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.subtract %arg0, %arg1
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.xor %arg0, %arg1
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @ZetaWithoutBroadcast
func @ZetaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> tensor<4xf32> {
  // CHECK: chlo.zeta %arg0, %arg1
  %0 = chlo.broadcast_zeta %arg0, %arg1
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @PolygammaWithoutBroadcast
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>)
func @PolygammaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> tensor<4xf32> {
  // CHECK: chlo.polygamma %[[LHS]], %[[RHS]]
  %0 = chlo.broadcast_polygamma %arg0, %arg1
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
