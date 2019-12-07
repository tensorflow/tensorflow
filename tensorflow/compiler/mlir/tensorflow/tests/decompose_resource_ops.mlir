// RUN: tf-opt %s -split-input-file -tf-device-decompose-resource-ops | FileCheck %s

// -----

// Tests that composite tf.AssignAddVariableOp operation is decomposed and
// hoisted.

// CHECK-LABEL: func @decompose_assign_add_variable_op
func @decompose_assign_add_variable_op() -> () {

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[ONE:[0-9]*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"
  // CHECK: "tf.AddV2"(%[[RES_READ_VAL]], %[[ONE]])
  // CHECK: "tf.AssignVariableOp"

  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  "tf.AssignAddVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<i32>) -> ()

  return
}

// -----

// Tests that composite tf.AssignSubVariableOp operation is decomposed using
// SubOp.

// CHECK-LABEL: func @decompose_assign_sub_variable_op
func @decompose_assign_sub_variable_op() -> () {

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[ONE:[0-9]*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"
  // CHECK: "tf.Sub"(%[[RES_READ_VAL]], %[[ONE]])
  // CHECK: "tf.AssignVariableOp"

  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  "tf.AssignSubVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<i32>) -> ()

  return
}

