// Note that binary elementwise tests are run with chlo legalization enabled
// (unlike the rest), since this is the primary use case for such ops and
// verification of shapes and broadcasts is desired.
// RUN: xla-opt "-xla-legalize-tf=allow-partial-conversion legalize-chlo=true" -canonicalize %s | FileCheck %s
// RUN: xla-opt "-xla-legalize-tf=allow-partial-conversion legalize-chlo=false" %s | FileCheck --check-prefix CHLO %s

//===----------------------------------------------------------------------===//
// Binary op legalizations.
// Most of these expand from the same pattern. Full semantics are
// verified for tf.Add and pattern application only for the rest.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @add
func.func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %[[SUM0:.*]] = mhlo.add %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %[[SUM0]] : tensor<2xi32>
  %1 = "tf.AddV2"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %1: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_add
// TODO(laurenzo): Change this to a (5 + 2x1) shaped add to make the check
// patterns unambiguous and more interesting (once broadcastable trait is
// fixed upstream).
func.func @broadcast_add(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: mhlo.add %[[LHS_BCAST]], %arg1
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  func.return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @broadcast_multi_dim_add
// TODO(laurenzo): Change this to a (4x1x1 + 1x4x4x4) shaped add once upstream
// broadcastable bug is fixed (helps make the CHECK matching unambiguous)
func.func @broadcast_multi_dim_add(%arg0: tensor<4x1x1xi32>, %arg1: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>}
  // CHECK-NEXT: mhlo.add %[[LHS_BCAST]], %arg1
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
  func.return %0: tensor<4x4x4x4xi32>
}

// CHECK-LABEL: func @add_dynamic
func.func @add_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG:  %[[CSTR_LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:  %[[CSTR_RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[CSTR_LHS_SHAPE]], %[[CSTR_RHS_SHAPE]]
  // CHECK-NEXT: shape.assuming %[[WITNESS:.+]]
  // CHECK-DAG:    %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:    %[[RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<1xindex>, tensor<2xindex> -> tensor<2xindex>
  // CHECK-NEXT:   %[[LHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT:   %[[RHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-NEXT:   %[[RESULT:.+]] = mhlo.add %[[LHS_BCAST]], %[[RHS_BCAST]] : tensor<?x?xi32>
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @broadcast_add_unranked
// CHLO-LABEL: func @broadcast_add_unranked
func.func @broadcast_add_unranked(%arg0: tensor<1xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: tf.Add
  // CHLO: chlo.broadcast_add %arg0, %arg1
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<1xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

// CHECK-LABEL: func @div
func.func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = mhlo.divide %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Div"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @shift_left
func.func @shift_left(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK:  mhlo.shift_left %arg0, %arg1 : tensor<4xi32>
  %0 = "tf.LeftShift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @div_unranked
func.func @div_unranked(%arg0: tensor<*xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-NEXT: tf.Div
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @maximum
func.func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:  mhlo.maximum %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @minimum
func.func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:  mhlo.minimum %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @mod
// CHLO-LABEL: func @mod
func.func @mod(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:  mhlo.remainder %arg0, %arg1 : tensor<4xf32>
  // CHLO: chlo.broadcast_remainder
  %0 = "tf.Mod"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @mul
func.func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = mhlo.multiply %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Mul"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @real_div
func.func @real_div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = mhlo.divide %arg0, %arg0 : tensor<2xi32>
  %0 = "tf.RealDiv"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @sub
func.func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = mhlo.subtract %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Sub"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @shift_right
func.func @shift_right(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK:  mhlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @shift_right_unsigned
func.func @shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // CHECK:  mhlo.shift_right_logical %arg0, %arg1 : tensor<4xui8>
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  func.return %0 : tensor<4xui8>
}

// CHECK-LABEL: func @broadcast_shift_right_unsigned
func.func @broadcast_shift_right_unsigned(%arg0: tensor<4xui8>, %arg1: tensor<2x4xui8>) -> tensor<2x4xui8> {
  // CHECK: %[[BROADCAST:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xui8>) -> tensor<2x4xui8>
  // CHECK:  mhlo.shift_right_logical %[[BROADCAST]], %arg1 : tensor<2x4xui8>
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4xui8>, tensor<2x4xui8>) -> tensor<2x4xui8>
  func.return %0 : tensor<2x4xui8>
}

// CHECK-LABEL: func @and
func.func @and(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.and
  %0 = "tf.LogicalAnd"(%arg0, %arg1) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @and_unranked
func.func @and_unranked(%arg0: tensor<*xi1>, %arg1: tensor<*xi1>) -> tensor<*xi1> {
  // CHECK: tf.LogicalAnd
  %0 = "tf.LogicalAnd"(%arg0, %arg1) : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @or
func.func @or(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.or
  %0 = "tf.LogicalOr"(%arg0, %arg1) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @bitwise_or
func.func @bitwise_or(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: mhlo.or
  %0 = "tf.BitwiseOr"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0: tensor<4xi32>
}

// CHECK-LABEL: func @bitwise_or_unsigned
func.func @bitwise_or_unsigned(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xui32> {
  // CHECK-NEXT: mhlo.or
  %0 = "tf.BitwiseOr"(%arg0, %arg1) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
  func.return %0: tensor<4xui32>
}

// CHECK-LABEL: func @bitwise_xor
func.func @bitwise_xor(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: mhlo.xor
  %0 = "tf.BitwiseXor"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0: tensor<4xi32>
}

// CHECK-LABEL: func @bitwise_xor_unsigned
func.func @bitwise_xor_unsigned(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xui32> {
  // CHECK-NEXT: mhlo.xor
  %0 = "tf.BitwiseXor"(%arg0, %arg1) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
  func.return %0: tensor<4xui32>
}

// CHECK-LABEL: func @bitwise_and
func.func @bitwise_and(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: mhlo.and
  %0 = "tf.BitwiseAnd"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0: tensor<4xi32>
}

// CHECK-LABEL: func @bitwise_and_unsigned
func.func @bitwise_and_unsigned(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xui32> {
  // CHECK-NEXT: mhlo.and
  %0 = "tf.BitwiseAnd"(%arg0, %arg1) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
  func.return %0: tensor<4xui32>
}

// CHECK-LABEL: func @pow
func.func @pow(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:  mhlo.power
  %0 = "tf.Pow"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0: tensor<2xf32>
}

//===----------------------------------------------------------------------===//
// Equality op legalizations.
// tf.Equal and tf.NotEqual expand from the same pattern. Full semantics are
// verified for tf.Equal and pattern application only for tf.NotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @equal
func.func @equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.compare EQ, %arg0, %arg1
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @equal_dynamic
func.func @equal_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  // TODO(jpienaar): Uncomment below when fallout from https://reviews.llvm.org/D83194 fixed.
  // NOT-CHECK-DAG:  %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // NOT-CHECK-DAG:  %[[RHS_SHAPE:.+]] = shape.const_shape [1]
  // NOT-CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[LHS_SHAPE]], %[[RHS_SHAPE]]
  // NOT-CHECK-NEXT: shape.assuming %[[WITNESS]] -> (tensor<?xi1>) {
  // NOT-CHECK-DAG:    %[[LHS_SHAPE1:.+]] = shape.shape_of %arg0
  // NOT-CHECK-NEXT:   %[[RESULT_SHAPE:.+]] = shape.broadcast %[[LHS_SHAPE1]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  // NOT-CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = tensor.cast %[[RESULT_SHAPE]] : tensor<?xindex> to tensor<1xindex>
  // NOT-CHECK-DAG:    %[[LHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // NOT-CHECK-DAG:    %[[RHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // NOT-CHECK-NEXT:   %[[RESULT:.+]] = mhlo.compare EQ, %[[LHS_BCAST]], %[[RHS_BCAST]]
  // NOT-CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  func.return %0: tensor<?xi1>
}

// CHECK-LABEL: func @equal_broadcast
func.func @equal_broadcast(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-DAG: %[[LHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: mhlo.compare EQ, %[[LHS_BCAST]], %arg1
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  func.return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @equal_broadcast_no_incompatible_shapes_error
func.func @equal_broadcast_no_incompatible_shapes_error(%arg0: tensor<2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = true}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  func.return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_broadcastable
func.func @equal_incompatible_shape_broadcastable(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = true}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  func.return %0: tensor<?xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_dynamic
func.func @equal_incompatible_shape_dynamic(%arg0: tensor<2xi32>, %arg1: tensor<?xi32>) -> tensor<*xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<?xi32>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @equal_incompatible_shape_both_dynamic
func.func @equal_incompatible_shape_both_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<*xi1> {
  // CHECK-NEXT: "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = false}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<?xi32>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @equal_unranked
func.func @equal_unranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi1> {
  // CHECK: "tf.Equal"
  // CHLO: chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #mhlo<comparison_direction EQ>}
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @equal_unsupported_type
func.func @equal_unsupported_type(%arg0: tensor<*x!tf_type.string>, %arg1: tensor<*x!tf_type.string>) -> tensor<*xi1> {
  // CHECK: "tf.Equal"
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<*x!tf_type.string>, tensor<*x!tf_type.string>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @notequal
func.func @notequal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.compare NE, %arg0, %arg1
  %0 = "tf.NotEqual"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

//===----------------------------------------------------------------------===//
// Compare op legalizations.
// These expand from the same pattern. Full semantics are checked for
// tf.Greater. Others just check that the pattern applied.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @greater
func.func @greater(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK: mhlo.compare GT, %arg0, %arg1
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_greater
func.func @broadcast_greater(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: %[[LHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: mhlo.compare GT, %[[LHS_BCAST]], %arg1
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  func.return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @greater_dynamic
func.func @greater_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  // CHECK-DAG:  %[[LHS_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK-DAG:  %[[RHS_SHAPE:.+]] = shape.shape_of %arg1
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[LHS_SHAPE]], %[[RHS_SHAPE]]
  // CHECK-NEXT: shape.assuming %[[WITNESS]]
  // CHECK-DAG:    %[[LHS_SHAPE1:.+]] = shape.shape_of %arg0
  // CHECK-DAG:    %[[RHS_SHAPE1:.+]] = shape.shape_of %arg1
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.broadcast %[[LHS_SHAPE1]], %[[RHS_SHAPE1]] : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex>
  // CHECK-DAG:    %[[LHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-DAG:    %[[RHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-NEXT:   mhlo.compare GT, %[[LHS_BCAST]], %[[RHS_BCAST]]
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0: tensor<?xi1>
}

// CHECK-LABEL: func @greater_uranked
func.func @greater_uranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi1> {
  // CHECK:  "tf.Greater"
  // CHLO: chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #mhlo<comparison_direction GT>}
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @greater_equal
func.func @greater_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.compare GE, %arg0, %arg1
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @less
func.func @less(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.compare LT, %arg0, %arg1
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @less_equal
func.func @less_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  mhlo.compare LE, %arg0, %arg1
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}
