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

// -----

// Tests that composite tf.ResourceApplyGradientDescent operation is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_gradient_descent
// CHECK-SAME: (%[[DELTA:.*]]: tensor<f32>)
func @decompose_resource_apply_gradient_descent(%arg0: tensor<f32>) -> () {

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[ALPHA:[0-9]*]] = "tf.Const"
  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  // CHECK: %[[MUL:[0-9]*]] = "tf.Mul"(%[[DELTA]], %[[ALPHA]])
  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]])
  // CHECK: %[[SUB:[0-9]*]] = "tf.Sub"(%[[RES_READ_VAL]], %[[MUL]])
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[SUB]])

  %1 = "tf.Const"() {T = f32, value = dense<[0.5]> : tensor<1xf32>} : () -> tensor<f32>
  "tf.ResourceApplyGradientDescent"(%0, %1, %arg0) {use_locking = false} : (tensor<*x!tf.resource>, tensor<f32>, tensor<f32>) -> ()

  return
}

// -----

// Tests that composite tf.ResourceApplyKerasMomentum (non-Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_keras_momentum_non_nesterov
// CHECK-SAME: (%[[LR:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>, %[[MOMENTUM:.*]]: tensor<f32>)
func @decompose_resource_apply_keras_momentum_non_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {

  // CHECK: %[[VAR_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  // CHECK: %[[ACCUM_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[ACCUM:[0-9]*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
  // CHECK: %[[ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[ACCUM]], %[[MOMENTUM]])
  // CHECK: %[[GRAD_LR:[0-9]*]] = "tf.Mul"(%[[GRAD]], %[[LR]])
  // CHECK: %[[NEW_ACCUM:[0-9]*]] = "tf.Sub"(%[[ACCUM_MOMENTUM]], %[[GRAD_LR]])
  // CHECK: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[NEW_ACCUM]])

  // CHECK: %[[VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]])
  // CHECK: %[[NEW_VAR:[0-9]*]] = "tf.AddV2"(%[[VAR]], %[[NEW_ACCUM]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[NEW_VAR]])

  "tf.ResourceApplyKerasMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = false} : (tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

  return
}

// -----

// Tests that composite tf.ResourceApplyKerasMomentum (with Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_keras_momentum_nesterov
// CHECK-SAME: (%[[LR:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>, %[[MOMENTUM:.*]]: tensor<f32>)
func @decompose_resource_apply_keras_momentum_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {

  // CHECK: %[[VAR_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  // CHECK: %[[ACCUM_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[ACCUM:[0-9]*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
  // CHECK: %[[ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[ACCUM]], %[[MOMENTUM]])
  // CHECK: %[[GRAD_LR:[0-9]*]] = "tf.Mul"(%[[GRAD]], %[[LR]])
  // CHECK: %[[NEW_ACCUM:[0-9]*]] = "tf.Sub"(%[[ACCUM_MOMENTUM]], %[[GRAD_LR]])
  // CHECK: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[NEW_ACCUM]])

  // CHECK: %[[NEW_ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[NEW_ACCUM]], %[[MOMENTUM]])
  // CHECK: %[[NEW_DELTA:[0-9]*]] = "tf.Sub"(%[[NEW_ACCUM_MOMENTUM]], %[[GRAD_LR]])
  // CHECK: %[[VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]])
  // CHECK: %[[NEW_VAR:[0-9]*]] = "tf.AddV2"(%[[VAR]], %[[NEW_DELTA]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[NEW_VAR]])

  "tf.ResourceApplyKerasMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = true} : (tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

  return
}

