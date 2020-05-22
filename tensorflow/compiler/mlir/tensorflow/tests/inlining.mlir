// RUN: tf-opt %s -inline="disable-simplify" | FileCheck %s --dump-input=fail

// Test that simple TF operations can be inlined.

func @inline_simple_callee() -> tensor<2xi32> {
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: func @inline_simple(
func @inline_simple() -> tensor<2xi32> {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: return %[[CST]]
  %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @inline_simple_callee} : () -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Check that TF call operations can be inlined, even when the shape of the
// argument or result is different than the called function.

func @inline_shape_cast_callee(%arg : tensor<*xi32>) -> tensor<*xi32> {
  return %arg : tensor<*xi32>
}

// CHECK-LABEL: func @inline_shape_cast(
// CHECK-SAME:                          %[[ARG:.*]]: tensor<2xi32>
func @inline_shape_cast(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: %[[ARG_CAST:.*]] = "tf.Cast"(%[[ARG]]) {Truncate = false} : (tensor<2xi32>) -> tensor<*xi32>
  // CHECK-NEXT: %[[RESULT_CAST:.*]] = "tf.Cast"(%[[ARG_CAST]]) {Truncate = false} : (tensor<*xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[RESULT_CAST]]
  %result = "tf.PartitionedCall"(%arg) {config = "", config_proto = "", executor_type = "", f = @inline_shape_cast_callee} : (tensor<2xi32>) -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Check that functions can be inlined into islands.

func @inline_into_island_multi_block_callee() -> tensor<2xi32> {
  br ^bb1

^bb1:
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: func @inline_into_island(
func @inline_into_island() -> (tensor<2xi32>, tensor<2xi32>) {
  %0:2 = tf_executor.graph {
    %1:3 = tf_executor.island {
      // Single block regions may be inlined.
      // CHECK: %[[CST:.*]] = "tf.Const"
      %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @inline_simple_callee} : () -> tensor<2xi32>

      // Multi block regions may not.
      // CHECK-NEXT: %[[CALL:.*]] = "tf.StatefulPartitionedCall"
      %result_2 = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @inline_into_island_multi_block_callee} : () -> tensor<2xi32>

      // CHECK-NEXT: tf_executor.yield %[[CST]], %[[CALL]]
      tf_executor.yield %result, %result_2 : tensor<2xi32>, tensor<2xi32>
    }
    tf_executor.fetch %1#1, %1#1 : tensor<2xi32>, tensor<2xi32>
  }
  return %0#1, %0#1 : tensor<2xi32>, tensor<2xi32>
}

