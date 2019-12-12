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

// -----

// Tests that composite tf.ResourceApplyAdam (non-Nesterov) operation is
// decomposed.

// CHECK-LABEL: func @decompose_resource_apply_adam_non_nesterov
// CHECK-SAME: ([[BETA1_POWER:%.*]]: tensor<f32>, [[BETA2_POWER:%.*]]: tensor<f32>, [[LR:%.*]]: tensor<f32>, [[BETA1:%.*]]: tensor<f32>, [[BETA2:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>)
func @decompose_resource_apply_adam_non_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>) -> () {

// CHECK: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
// CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"()
// CHECK: [[M_HANDLE:%.*]] = "tf.VarHandleOp"()
// CHECK: [[V_HANDLE:%.*]] = "tf.VarHandleOp"()
// CHECK: [[ONE_MINUS_BETA2_POWER:%.*]] = "tf.Sub"([[ONE]], [[BETA2_POWER]])
// CHECK: [[SQRT_ONE_MINUS_BETA2_POWER:%.*]] = "tf.Sqrt"([[ONE_MINUS_BETA2_POWER]])
// CHECK: [[ONE_MINUS_BETA1_POWER:%.*]] = "tf.Sub"([[ONE]], [[BETA1_POWER]])
// CHECK: [[ALPHA_NO_LR:%.*]] = "tf.Div"([[SQRT_ONE_MINUS_BETA2_POWER]], [[ONE_MINUS_BETA1_POWER]])
// CHECK: [[ALPHA:%.*]] = "tf.Mul"([[LR]], [[ALPHA_NO_LR]])
// CHECK: [[OLD_M:%.*]] = "tf.ReadVariableOp"([[M_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[BETA1_OLD_M:%.*]] = "tf.Mul"([[BETA1]], [[OLD_M]])
// CHECK: [[ONE_MINUS_BETA1:%.*]] = "tf.Sub"([[ONE]], [[BETA1]])
// CHECK: [[ONE_MINUS_BETA1_GRAD:%.*]] = "tf.Mul"([[ONE_MINUS_BETA1]], [[GRAD]])
// CHECK: [[NEW_M:%.*]] = "tf.AddV2"([[BETA1_OLD_M]], [[ONE_MINUS_BETA1_GRAD]])
// CHECK: [[OLD_V:%.*]] = "tf.ReadVariableOp"([[V_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[BETA2_OLD_V:%.*]] = "tf.Mul"([[BETA2]], [[OLD_V]])
// CHECK: [[ONE_MINUS_BETA2:%.*]] = "tf.Sub"([[ONE]], [[BETA2]])
// CHECK: [[GRAD_SQUARE:%.*]] = "tf.Square"([[GRAD]])
// CHECK: [[V_DELTA:%.*]] = "tf.Mul"([[ONE_MINUS_BETA2]], [[GRAD_SQUARE]])
// CHECK: [[NEW_V:%.*]] = "tf.AddV2"([[BETA2_OLD_V]], [[V_DELTA]])
// CHECK: [[ALPHA_NEW_M:%.*]] = "tf.Mul"([[ALPHA]], [[NEW_M]])
// CHECK: [[SQRT_NEW_V:%.*]] = "tf.Sqrt"([[NEW_V]])
// CHECK: [[SQRT_NEW_V_EPSILON:%.*]] = "tf.AddV2"([[SQRT_NEW_V]], [[EPSILON]])
// CHECK: [[VAR_DELTA:%.*]] = "tf.Div"([[ALPHA_NEW_M]], [[SQRT_NEW_V_EPSILON]])
// CHECK: [[OLD_VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[NEW_VAR:%.*]] = "tf.Sub"([[OLD_VAR]], [[VAR_DELTA]])
// CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[NEW_VAR]])
// CHECK: "tf.AssignVariableOp"([[M_HANDLE]], [[NEW_M]])
// CHECK: "tf.AssignVariableOp"([[V_HANDLE]], [[NEW_V]])

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  "tf.ResourceApplyAdam"(%0, %1, %2, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {use_locking = false, use_nesterov = false} : (tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

  return
}

// -----

// Tests that composite tf.ResourceApplyAdam (with Nesterov) operation is
// decomposed.

// CHECK-LABEL: func @decompose_resource_apply_adam_nesterov(
// CHECK-SAME:  [[BETA1_POWER:%.*]]: tensor<f32>, [[BETA2_POWER:%.*]]: tensor<f32>, [[LR:%.*]]: tensor<f32>, [[BETA1:%.*]]: tensor<f32>, [[BETA2:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>) {
func @decompose_resource_apply_adam_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>) -> () {

// CHECK: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
// CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
// CHECK: [[M_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
// CHECK: [[V_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
// CHECK: [[VAL_82:%.*]] = "tf.Sub"([[ONE]], [[BETA2_POWER]])
// CHECK: [[VAL_83:%.*]] = "tf.Sqrt"([[VAL_82]])
// CHECK: [[VAL_84:%.*]] = "tf.Sub"([[ONE]], [[BETA1_POWER]])
// CHECK: [[VAL_85:%.*]] = "tf.Div"([[VAL_83]], [[VAL_84]])
// CHECK: [[VAL_86:%.*]] = "tf.Mul"([[LR]], [[VAL_85]])
// CHECK: [[OLD_M:%.*]] = "tf.ReadVariableOp"([[M_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[VAL_88:%.*]] = "tf.Mul"([[BETA1]], [[OLD_M]])
// CHECK: [[VAL_89:%.*]] = "tf.Sub"([[ONE]], [[BETA1]])
// CHECK: [[VAL_90:%.*]] = "tf.Mul"([[VAL_89]], [[GRAD]])
// CHECK: [[NEW_M:%.*]] = "tf.AddV2"([[VAL_88]], [[VAL_90]])
// CHECK: [[OLD_V:%.*]] = "tf.ReadVariableOp"([[V_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[VAL_93:%.*]] = "tf.Mul"([[BETA2]], [[OLD_V]])
// CHECK: [[VAL_94:%.*]] = "tf.Sub"([[ONE]], [[BETA2]])
// CHECK: [[VAL_95:%.*]] = "tf.Square"([[GRAD]])
// CHECK: [[VAL_96:%.*]] = "tf.Mul"([[VAL_94]], [[VAL_95]])
// CHECK: [[NEW_V:%.*]] = "tf.AddV2"([[VAL_93]], [[VAL_96]])
// CHECK: [[VAL_98:%.*]] = "tf.Mul"([[NEW_M]], [[BETA1]])
// CHECK: [[VAL_99:%.*]] = "tf.Sub"([[ONE]], [[BETA1]])
// CHECK: [[VAL_100:%.*]] = "tf.Mul"([[VAL_99]], [[GRAD]])
// CHECK: [[VAL_101:%.*]] = "tf.AddV2"([[VAL_98]], [[VAL_100]])
// CHECK: [[VAL_102:%.*]] = "tf.Mul"([[VAL_86]], [[VAL_101]])
// CHECK: [[VAL_103:%.*]] = "tf.Sqrt"([[NEW_V]])
// CHECK: [[VAL_104:%.*]] = "tf.AddV2"([[VAL_103]], [[EPSILON]])
// CHECK: [[VAL_105:%.*]] = "tf.Div"([[VAL_102]], [[VAL_104]])
// CHECK: [[OLD_VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]]) : (tensor<*x!tf.resource>) -> tensor<*xf32>
// CHECK: [[NEW_VAR:%.*]] = "tf.Sub"([[OLD_VAR]], [[VAL_105]])
// CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[NEW_VAR]]) : (tensor<*x!tf.resource>, tensor<*xf32>) -> ()
// CHECK: "tf.AssignVariableOp"([[M_HANDLE]], [[NEW_M]]) : (tensor<*x!tf.resource>, tensor<*xf32>) -> ()
// CHECK: "tf.AssignVariableOp"([[V_HANDLE]], [[NEW_V]]) : (tensor<*x!tf.resource>, tensor<*xf32>) -> ()

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>
  %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  "tf.ResourceApplyAdam"(%0, %1, %2, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {use_locking = false, use_nesterov = true} : (tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

  return
}
