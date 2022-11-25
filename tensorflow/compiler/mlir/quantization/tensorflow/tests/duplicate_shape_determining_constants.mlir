// RUN: tf-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -quant-duplicate-shape-determining-constants | FileCheck %s

// CHECK-LABEL: @duplicate_const_for_shape_determining_operand_at_idx_1
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?x2xi32>)
func.func private @duplicate_const_for_shape_determining_operand_at_idx_1(%arg0: tensor<?x2xi32>) -> tensor<?x2x1xi32> {
  %cst = "tf.Const"() {device = "", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // idx 1 should be a compile time constant
  %0 = "tf.ExpandDims"(%arg0, %cst) {device = ""} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?x2x1xi32>
  %1 = "tf.AddV2"(%cst, %cst) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %0 : tensor<?x2x1xi32>
}
// Check that the constant is cloned with same value.
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<2> : tensor<i32>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<2> : tensor<i32>

// Check that the constants used for tf.ExpandDims and tf.AddV2 are different.
// CHECK: %[[EXPAND_DIMS:.*]] = "tf.ExpandDims"(%[[ARG_0]], %[[CST_1]])
// CHECK: %[[ADDV2:.*]] = "tf.AddV2"(%[[CST_0]], %[[CST_0]])

// -----

// CHECK-LABEL: @duplicate_const_for_shape_determining_operand_at_idx_2
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<16x4xf32>, %[[ARG_1:.*]]: tensor<16xi32>)
func.func private @duplicate_const_for_shape_determining_operand_at_idx_2(%arg0: tensor<16x4xf32>, %arg1: tensor<16xi32>) -> tensor<16xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  // idx 2 should be a compile time constant
  %0 = "tf.GatherV2"(%arg0, %arg1, %cst) {batch_dims = 1: i64} : (tensor<16x4xf32>, tensor<16xi32>, tensor<1xi32>) -> tensor<16xf32>

  // Just to introduce an extra use for %cst.
  %1 = "tf.AddV2"(%cst, %cst) {device = ""} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  return %0 : tensor<16xf32>
}
// Check that the constant is cloned with same value.
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<1xi32>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<1xi32>

// Check that the constants used for tf.GatherV2 and tf.AddV2 are different.
// CHECK: %[[GATHER_V2:.*]] = "tf.GatherV2"(%[[ARG_0]], %[[ARG_1]], %[[CST_1]])
// CHECK: %[[ADDV2:.*]] = "tf.AddV2"(%[[CST_0]], %[[CST_0]])

// -----

// CHECK-LABEL: @duplicate_const_for_shape_determining_operand_with_variadic_operand
// CHECK-SAME: %[[ARG_0:.*]]: tensor<16x1xf32>
func.func private @duplicate_const_for_shape_determining_operand_with_variadic_operand(%arg0: tensor<16x1xf32>) -> tensor<16x4xf32> {
  %axis = "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // tf.ConcatV2 accepts a variadic operand. The last operand should be compile
  // time constant.
  %0 = "tf.ConcatV2"(%arg0, %arg0, %arg0, %arg0, %axis) : (tensor<16x1xf32>, tensor<16x1xf32>, tensor<16x1xf32>, tensor<16x1xf32>, tensor<i32>) -> tensor<16x4xf32>

  // Just to introduce an extra use for %cst.
  %1 = "tf.AddV2"(%axis, %axis) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %0 : tensor<16x4xf32>
}
// Check that the constant is cloned with same value.
// The duplicated constant is the last index of the ConcatV2 op (which
// accepts a variadic arg).
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<i32>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<i32>

// Check that the constants used for tf.ConcatV2 and tf.AddV2 are different.
// CHECK: %[[CONCAT_V2:.*]] = "tf.ConcatV2"(%[[ARG_0]], %[[ARG_0]], %[[ARG_0]], %[[ARG_0]], %[[CST_1]])
// CHECK: %[[ADDV2:.*]] = "tf.AddV2"(%[[CST_0]], %[[CST_0]])

// -----

// CHECK-LABEL: @duplicate_const_for_multiple_shape_determining_operands
// CHECK-SAME: %[[ARG_0:.*]]: tensor<8x4x16x16x16xf32>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<4x3x3x16x16xf32>
func.func private @duplicate_const_for_multiple_shape_determining_operands(
    %arg0: tensor<8x4x16x16x16xf32>, %arg1: tensor<4x3x3x16x16xf32>) -> tensor<8x4x14x14x16xf32> {
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %lhs_dilation = "tf.Const"() {value = dense<[4, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

  // tf.XlaConvV2's 2, 3, 4, 5, 6 indices should be compile-time constants.
  %0 = "tf.XlaConvV2"(%arg0, %arg1, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {
      batch_group_count = 1 : i64,
      dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03",
      precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>,
         tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>

  // Just to introduce an extra use for %cst.
  %1 = "tf.AddV2"(%feature_group_count, %feature_group_count) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%lhs_dilation, %lhs_dilation) {device = ""} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  %3 = "tf.AddV2"(%rhs_dilation, %rhs_dilation) {device = ""} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  %4 = "tf.AddV2"(%padding, %padding) {device = ""} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  %5 = "tf.AddV2"(%strides, %strides) {device = ""} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>

  return %0 : tensor<8x4x14x14x16xf32>
}

// Check that the constants that are input to XlaConvV2's 3rd, 4th, 5th, 6th
// and 7th arguments are cloned with same value.
// CHECK-DAG: %[[STRIDES:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<[3, 1, 1]> : tensor<3xi32>
// CHECK-DAG: %[[STRIDES_COPY:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<[3, 1, 1]> : tensor<3xi32>
// CHECK-DAG: %[[PADDING:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<0> : tensor<3x2xi32>
// CHECK-DAG: %[[PADDING_COPY:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<0> : tensor<3x2xi32>
// CHECK-DAG: %[[LHS_DILATION:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<[4, 1, 1]> : tensor<3xi32>
// CHECK-DAG: %[[LHS_DILATION_COPY:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<[4, 1, 1]> : tensor<3xi32>
// CHECK-DAG: %[[RHS_DILATION:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<3xi32>
// CHECK-DAG: %[[RHS_DILATION_COPY:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<3xi32>
// CHECK-DAG: %[[FEATURE_GROUP_COUNT:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<i32>
// CHECK-DAG: %[[FEATURE_GROUP_COUNT_COPY:.*]] = "tf.Const"()
// CHECK-SAME: value = dense<1> : tensor<i32>

// Check that the constants that are input to XlaConvV2's 3rd and 4th
// arguments are not duplicated.
// CHECK-NOT: "tf.Const"()

// Check that the constants used for tf.XlaConvV2 and tf.AddV2s are different.
// CHECK: %[[GATHER_V2:.*]] = "tf.XlaConvV2"(%[[ARG_0]], %[[ARG_1]], %[[STRIDES_COPY]], %[[PADDING_COPY]], %[[LHS_DILATION_COPY]], %[[RHS_DILATION_COPY]], %[[FEATURE_GROUP_COUNT_COPY]])

// CHECK: %[[ADDV2_2:.*]] = "tf.AddV2"(%[[FEATURE_GROUP_COUNT]], %[[FEATURE_GROUP_COUNT]])
// CHECK: %[[ADDV2_0:.*]] = "tf.AddV2"(%[[LHS_DILATION]], %[[LHS_DILATION]])
// CHECK: %[[ADDV2_1:.*]] = "tf.AddV2"(%[[RHS_DILATION]], %[[RHS_DILATION]])

// -----

// CHECK-LABEL: @stop_recursion_when_arg_is_reached
func.func private @stop_recursion_when_arg_is_reached(%arg0: tensor<1x2x3xf32>, %arg1: tensor<i32>) -> tensor<?x?x?xf32> {
// The pass wants to duplicate constants for TF::MeanOp's operand idx 1, but
// it can't proceed since it is a function argument.

// expected-warning @+1 {{Operand idx (zero-based): 1 does not have a defining op and cannot be duplicated}}
  %0 = "tf.Mean"(%arg0, %arg1) {device = ""} : (tensor<1x2x3xf32>, tensor<i32>) -> tensor<?x?x?xf32>

  return %0: tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @constant_with_single_use_not_duplicated
func.func private @constant_with_single_use_not_duplicated(%arg0: tensor<1x2x3xf32>) -> tensor<1x3xf32> {
  %cst = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%cst, %cst_0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Max"(%arg0, %0) {device = ""} : (tensor<1x2x3xf32>, tensor<i32>) -> tensor<1x3xf32>

  return %1: tensor<1x3xf32>
}
// CHECK-DAG: %[[CST:.*]] = "tf.Const"
// CHECK-SAME: dense<0>
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"
// CHECK-SAME: dense<1>
// Check that there are no extra "tf.Const"s existing in this function.
// CHECK-NOT: "tf.Const"

// Check that the usages of %[[CST]] and %[[CST_0]] are untouched.
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[CST]], %[[CST_0]])
// CHECK: "tf.Max"({{.*}}, %[[ADD]])

// -----

// CHECK-LABEL: @recursively_duplicate_constants
func.func private @recursively_duplicate_constants(%arg0: tensor<1x2x3xf32>) -> tensor<1x3xf32> {
  %cst = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%cst, %cst_0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Max"(%arg0, %0) {device = ""} : (tensor<1x2x3xf32>, tensor<i32>) -> tensor<1x3xf32>

  // Just to introduce extra usages for %cst and %cst_0.
  %2 = "tf.Mul"(%cst, %cst_0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %1: tensor<1x3xf32>
}
// Check that both constants are duplicated, which are used to transitively
// determine the shape of the result of `tf.Max`.
// CHECK-DAG: %[[CST:.*]] = "tf.Const"
// CHECK-SAME: dense<0>
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"
// CHECK-SAME: dense<0>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"
// CHECK-SAME: dense<1>
// CHECK-DAG: %[[CST_2:.*]] = "tf.Const"
// CHECK-SAME: dense<1>

// -----

// CHECK-LABEL: @early_stop_at_shape_op
func.func private @early_stop_at_shape_op() -> tensor<1x3xi32> {
  %cst = "tf.Const"() {device = "", value = dense<1.0> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Shape"(%cst) : (tensor<1x3xf32>) -> tensor<2xi32>
  // Operand index 0 ($dims) should be a compile-time constant.
  %2 = "tf.Fill"(%1, %cst_0) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<1x3xi32>

  // Just to introduce extra usages for %cst.
  %3 = "tf.Mul"(%cst, %cst) {device = ""} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>

  return %2: tensor<1x3xi32>
}
// The output of tf.Shape is considered a compile-time constant, so the
// constant leading to tf.Shape (which transitively becomes an input to the
// first arg of tf.Fill) is not duplicated.

// CHECK-DAG: %[[CST:.*]] = "tf.Const"
// CHECK-SAME: dense<1.000000e+00> : tensor<1x3xf32>
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"
// CHECK-SAME: dense<2> : tensor<i32>
// CHECK: %[[SHAPE:.*]] = "tf.Shape"(%[[CST]])
// CHECK: %[[FILL:.*]] = "tf.Fill"(%[[SHAPE]], %[[CST_0]])
