// Note that binary elementwise tests are run with chlo legalization enabled
// (unlike the rest), since this is the primary use case for such ops and
// verification of shapes and broadcasts is desired.
// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion legalize-chlo=true" -canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Binary op legalizations.
// Most of these expand from the same pattern. Full semantics are
// verified for tf.Add and pattern application only for the rest.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @add
func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %[[SUM0:.*]] = xla_hlo.add %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  %[[SUM1:.*]] = xla_hlo.add %[[SUM0]], %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %[[SUM1]] : tensor<2xi32>
  %0 = "tf.Add"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tf.AddV2"(%0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %1: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_add
// TODO(laurenzo): Change this to a (5 + 2x1) shaped add to make the check
// patterns unambiguous and more interesting (once broadcastable trait is
// fixed upstream).
func @broadcast_add(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: xla_hlo.add %[[LHS_BCAST]], %arg1
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @broadcast_multi_dim_add
// TODO(laurenzo): Change this to a (4x1x1 + 1x4x4x4) shaped add once upstream
// broadcastable bug is fixed (helps make the CHECK matching unambiguous)
func @broadcast_multi_dim_add(%arg0: tensor<4x1x1xi32>, %arg1: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>}
  // CHECK-NEXT: xla_hlo.add %[[LHS_BCAST]], %arg1
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
  return %0: tensor<4x4x4x4xi32>
}

// CHECK-LABEL: func @add_dynamic
func @add_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG:  %[[CSTR_LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:  %[[CSTR_RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[CSTR_LHS_SHAPE]], %[[CSTR_RHS_SHAPE]]
  // CHECK-NEXT: shape.assuming %[[WITNESS:.+]]
  // CHECK-DAG:    %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:    %[[RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT:   %[[RESULT_SHAPE:.+]] = "shape.broadcast"(%[[LHS_SHAPE]], %[[RHS_SHAPE]])
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_SHAPE]]
  // CHECK-NEXT:   %[[LHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT:   %[[RHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-NEXT:   %[[RESULT:.+]] = xla_hlo.add %[[LHS_BCAST]], %[[RHS_BCAST]] : tensor<?x?xi32>
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @div
func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Div"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @shift_left
func @shift_left(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK:  xla_hlo.shift_left %arg0, %arg1 : tensor<4xi32>
  %0 = "tf.LeftShift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @div_unranked
func @div_unranked(%arg0: tensor<*xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-NEXT: tf.Div
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @maximum
func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:  xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @minimum
func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:  xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @mul
func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.multiply %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Mul"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @real_div
func @real_div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
  %0 = "tf.RealDiv"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @sub
func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.subtract %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Sub"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @shift_right
func @shift_right(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK:  xla_hlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @shift_right_unsigned
func @shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // CHECK:  tf.RightShift
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}

// CHECK-LABEL: func @broadcast_shift_right_unsigned
func @broadcast_shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<2x4xui8>) -> tensor<2x4xui8> {
  // CHECK:  tf.RightShift
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<2x4xui8>) -> tensor<2x4xui8>
  return %0 : tensor<2x4xui8>
}

// CHECK-LABEL: func @and
func @and(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT:  xla_hlo.and
  %0 = "tf.LogicalAnd"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @and_unranked
func @and_unranked(%arg0: tensor<*xi1>, %arg1: tensor<*xi1>) -> tensor<*xi1> {
  // CHECK: tf.LogicalAnd
  %0 = "tf.LogicalAnd"(%arg0, %arg1) : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  return %0: tensor<*xi1>
}

// CHECK-LABEL: func @or
func @or(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT:  xla_hlo.or
  %0 = "tf.LogicalOr"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @bitwise_or
func @bitwise_or(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: xla_hlo.or
  %0 = "tf.BitwiseOr"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0: tensor<4xi32>
}

// CHECK-LABEL: func @bitwise_and
func @bitwise_and(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: xla_hlo.and
  %0 = "tf.BitwiseAnd"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0: tensor<4xi32>
}

// CHECK-LABEL: func @pow
func @pow(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:  xla_hlo.power
  %0 = "tf.Pow"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0: tensor<2xf32>
}

//===----------------------------------------------------------------------===//
// Equality op legalizations.
// tf.Equal and tf.NotEqual expand from the same pattern. Full semantics are
// verified for tf.Equal and pattern application only for tf.NotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @equal
func @equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"}
  %0 = "tf.Equal"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @equal_dynamic
func @equal_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  // CHECK-DAG:  %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:  %[[RHS_SHAPE:.+]] = shape.const_shape [1]
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[LHS_SHAPE]], %[[RHS_SHAPE]]
  // CHECK-NEXT: shape.assuming %[[WITNESS]] -> (tensor<?xi1>) {
  // CHECK-DAG:    %[[LHS_SHAPE1:.+]] = shape.shape_of %arg0
  // CHECK-NEXT:   %[[RESULT_SHAPE:.+]] = "shape.broadcast"(%[[LHS_SHAPE1]], %[[RHS_SHAPE]])
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_SHAPE]]
  // CHECK-DAG:    %[[LHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-DAG:    %[[RHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-NEXT:   %[[RESULT:.+]] = "xla_hlo.compare"(%[[LHS_BCAST]], %[[RHS_BCAST]]) {comparison_direction = "EQ"}
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  return %0: tensor<?xi1>
}

// CHECK-LABEL: func @equal_broadcast
func @equal_broadcast(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-DAG: %[[LHS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: "xla_hlo.compare"(%[[LHS_BCAST]], %arg1) {comparison_direction = "EQ"}
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @equal_broadcast_no_incompatible_shapes_error
func @equal_broadcast_no_incompatible_shapes_error(%arg0: tensor<2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_broadcastable
func @equal_incompatible_shape_broadcastable(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  return %0: tensor<?xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_dynamic
func @equal_incompatible_shape_dynamic(%arg0: tensor<2xi32>, %arg1: tensor<?xi32>) -> tensor<*xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<?xi32>) -> tensor<*xi1>
  return %0: tensor<*xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_both_dynamic
func @equal_incompatible_shape_both_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<*xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<?xi32>) -> tensor<*xi1>
  return %0: tensor<*xi1>
}

// CHECK-LABEL: func @equal_unranked
func @equal_unranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi1> {
  // CHECK: "tf.Equal"
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
  return %0: tensor<*xi1>
}

// CHECK-LABEL: func @notequal
func @notequal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"}
  %0 = "tf.NotEqual"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

//===----------------------------------------------------------------------===//
// Compare op legalizations.
// These expand from the same pattern. Full semantics are checked for
// tf.Greater. Others just check that the pattern applied.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @greater
func @greater(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK: "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GT"}
  %0 = "tf.Greater"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_greater
func @broadcast_greater(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: "xla_hlo.compare"(%[[LHS_BCAST]], %arg1) {comparison_direction = "GT"}
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @greater_dynamic
func @greater_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  // CHECK-DAG:  %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:  %[[RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[LHS_SHAPE]], %[[RHS_SHAPE]]
  // CHECK-NEXT: shape.assuming %[[WITNESS]]
  // CHECK-DAG:    %[[LHS_SHAPE1:.+]] = shape.shape_of %arg0
  // CHECK-DAG:    %[[RHS_SHAPE1:.+]] = shape.shape_of %arg1
  // CHECK-NEXT:   %[[RESULT_SHAPE:.+]] = "shape.broadcast"(%[[LHS_SHAPE1]], %[[RHS_SHAPE1]])
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_SHAPE]]
  // CHECK-DAG:    %[[LHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-DAG:    %[[RHS_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-NEXT:   "xla_hlo.compare"(%[[LHS_BCAST]], %[[RHS_BCAST]]) {comparison_direction = "GT"}
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  return %0: tensor<?xi1>
}

// CHECK-LABEL: func @greater_uranked
func @greater_uranked(%arg0: tensor<*xi32>) -> tensor<*xi1> {
  // CHECK:  "tf.Greater"
  %0 = "tf.Greater"(%arg0, %arg0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
  return %0: tensor<*xi1>
}

// CHECK-LABEL: func @greater_equal
func @greater_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GE"}
  %0 = "tf.GreaterEqual"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @less
func @less(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT"}
  %0 = "tf.Less"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @less_equal
func @less_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LE"}
  %0 = "tf.LessEqual"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}
