// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-promote-var-handles-to-args | FileCheck %s -dump-input-on-failure

// Tests main function with multiple blocks.

// expected-error@+1 {{expects function 'main' to have 1 block, got 2}}
func @main() {
  br ^bb1
^bb1:
  return
}

// -----

// CHECK-LABEL: func @no_args
// CHECK-SAME: (%arg0: tensor<!tf.resource> {tf.resource_name = "x"})
// CHECK-NOT: "tf.VarHandleOp"
func @no_args() {
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource>
  return
}

// CHECK-LABEL: func @some_args
// CHECK-SAME: (%arg0: tensor<i1>, %arg1: tensor<!tf.resource> {tf.resource_name = "x"})
// CHECK-NOT: "tf.VarHandleOp"
func @some_args(%arg0: tensor<i1>) {
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource>
  return
}

// CHECK-LABEL: func @unique_vars
// CHECK-SAME: (%arg0: tensor<!tf.resource<tensor<f32>>> {tf.resource_name = "x"}, %arg1: tensor<!tf.resource<tensor<i32>>> {tf.resource_name = "y"})
// CHECK-NOT: "tf.VarHandleOp"
func @unique_vars() {
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "y"} : () -> tensor<!tf.resource<tensor<i32>>>
  return
}

// CHECK-LABEL: func @duplicate_vars
// CHECK-SAME: (%arg0: tensor<!tf.resource<tensor<f32>>> {tf.resource_name = "x"})
// CHECK-NOT: "tf.VarHandleOp"
func @duplicate_vars() {
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  return
}

// CHECK-LABEL: func @duplicate_vars_with_users
// CHECK-SAME: (%arg0: tensor<f32>, %arg1: tensor<!tf.resource<tensor<f32>>> {tf.resource_name = "x"})
// CHECK: "tf.ReadVariableOp"(%arg1)
// CHECK: "tf.AssignAddVariableOp"(%arg1, %arg0)
// CHECK-NOT: "tf.VarHandleOp"
func @duplicate_vars_with_users(%arg0: tensor<f32>) {
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %2 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  "tf.AssignAddVariableOp"(%2, %arg0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}
