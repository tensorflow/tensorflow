// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @remove_op_with_inner_ops_pattern
func @remove_op_with_inner_ops_pattern() {
  // CHECK-NEXT: return
  "test.op_with_region_pattern"() ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold_no_side_effect
func @remove_op_with_inner_ops_fold_no_side_effect() {
  // CHECK-NEXT: return
  "test.op_with_region_fold_no_side_effect"() ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func @remove_op_with_inner_ops_fold(%arg0 : i32) -> (i32) {
  // CHECK-NEXT: return %[[ARG_0]]
  %0 = "test.op_with_region_fold"(%arg0) ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : (i32) -> (i32)
  return %0 : i32
}

// CHECK-LABEL: func @remove_op_with_variadic_results_and_folder
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func @remove_op_with_variadic_results_and_folder(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK-NEXT: return %[[ARG_0]], %[[ARG_1]]
  %0, %1 = "test.op_with_variadic_results_and_folder"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
