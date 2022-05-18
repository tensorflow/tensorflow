// RUN: tf-opt %s -tf-device-decompose-resource-ops | FileCheck -check-prefixes=CHECK,ALWAYS-DECOMPOSE %s
// RUN: tf-opt %s -tf-device-decompose-resource-ops-in-cluster | FileCheck -check-prefixes=CHECK,CLUSTER-ONLY %s

// Tests different behavior of two passes if the op is outside cluster
// tf-device-decompose-resource-ops would always decompose the resource op.
// tf-device-decompose-resource-ops-in-cluster would not decompose ops outside
// cluster or functions transitively called from within clusters.

// CHECK-LABEL: func @decomposition_outside_cluster
func.func @decomposition_outside_cluster() {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8xi32>>>
  // CHECK:      %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CLUSTER-ONLY: "tf.AssignAddVariableOp"
  // ALWAYS-DECOMPOSE-NOT:  "tf.AssignAddVariableOp"
  "tf.AssignAddVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
  // CHECK: tf_device.cluster
  "tf_device.cluster"() ({
    // CHECK-NOT: "tf.AssignAddVariableOp"
    "tf.AssignAddVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// Test the behavior of passes on ops in reachable/unreachable functions from
// cluster.
// tf-device-decompose-resource-ops would always decompose the resource op.
// tf-device-decompose-resource-ops-in-cluster would decompose ops in reachable
// function but not in unreachable function.


// CHECK-LABEL: func @decomposition_in_referenced_functions
func.func @decomposition_in_referenced_functions() {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8xi32>>>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  "tf_device.cluster"() ({
    "tf.OpA"(%0, %1) {f = @reachable_function} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// CHECK-LABEL: func @reachable_function
func.func @reachable_function(%arg0: tensor<*x!tf_type.resource<tensor<2x8xi32>>>, %arg1: tensor<i32>) {
  // CHECK-NOT: "tf.AssignAddVariableOp"
  "tf.AssignAddVariableOp"(%arg0, %arg1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
  "tf.OpA"(%arg0, %arg1) {f = @reachable_from_reachable_function} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @reachable_from_reachable_function
func.func @reachable_from_reachable_function(%arg0: tensor<*x!tf_type.resource<tensor<2x8xi32>>>, %arg1: tensor<i32>) {
  // CHECK-NOT: "tf.AssignAddVariableOp"
  "tf.AssignAddVariableOp"(%arg0, %arg1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @unreachable_function
func.func @unreachable_function(%arg0: tensor<*x!tf_type.resource<tensor<2x8xi32>>>, %arg1: tensor<i32>) {
  // CLUSTER-ONLY: "tf.AssignAddVariableOp"
  // ALWAYS-DECOMPOSE-NOT:  "tf.AssignAddVariableOp"
  "tf.AssignAddVariableOp"(%arg0, %arg1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
  func.return
}

// The tests below have the same behavior for the two passes as all ops are
// within device cluster.

// Tests that resources with subtypes are used if present.

// CHECK-LABEL: func @decompose_use_subtype
func.func @decompose_use_subtype() {
  "tf_device.cluster"() ({
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8xi32>>>

    // CHECK:      %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK:      %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"
    // CHECK-SAME: (tensor<*x!tf_type.resource<tensor<2x8xi32>>>) -> tensor<2x8xi32>
    // CHECK:      "tf.AddV2"(%[[RES_READ_VAL]], %[[ONE]])
    // CHECK-SAME: (tensor<2x8xi32>, tensor<i32>) -> tensor<2x8xi32>
    // CHECK:      "tf.AssignVariableOp"

    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    tf_device.return
  }) : () -> ()

  func.return
}

// Tests that composite tf.AssignAddVariableOp operation is decomposed and
// hoisted.

// CHECK-LABEL: func @decompose_assign_add_variable_op
func.func @decompose_assign_add_variable_op() -> () {
  "tf_device.cluster"() ({

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<i32>>>

    // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"
    // CHECK: "tf.AddV2"(%[[RES_READ_VAL]], %[[ONE]])
    // CHECK: "tf.AssignVariableOp"

    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.AssignSubVariableOp operation is decomposed using
// SubOp.

// CHECK-LABEL: func @decompose_assign_sub_variable_op
func.func @decompose_assign_sub_variable_op() -> () {
  "tf_device.cluster"() ({

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<i32>>>

    // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"
    // CHECK: "tf.Sub"(%[[RES_READ_VAL]], %[[ONE]])
    // CHECK: "tf.AssignVariableOp"

    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.AssignSubVariableOp"(%0, %1) {dtype = "tfdtype$DT_INT32"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyGradientDescent operation is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_gradient_descent
// CHECK-SAME: (%[[DELTA:.*]]: tensor<f32>)
func.func @decompose_resource_apply_gradient_descent(%arg0: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<f32>>>

    // CHECK: %[[ALPHA:.*]] = "tf.Const"
    // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]])
    // CHECK: %[[MUL:[0-9]*]] = "tf.Mul"(%[[DELTA]], %[[ALPHA]])
    // CHECK: %[[SUB:[0-9]*]] = "tf.Sub"(%[[RES_READ_VAL]], %[[MUL]])
    // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[SUB]])

    %1 = "tf.Const"() {T = f32, value = dense<[0.5]> : tensor<1xf32>} : () -> tensor<f32>
    "tf.ResourceApplyGradientDescent"(%0, %1, %arg0) {use_locking = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyMomentum (non-Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_momentum_non_nesterov
// CHECK-SAME:  [[LR:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>, [[MOMENTUM:%.*]]: tensor<f32>
func.func @decompose_resource_apply_momentum_non_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"
    // CHECK: [[ACCUM_HANDLE:%.*]] = "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<f32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<f32>>>

    // CHECK: [[ACCUM:%.*]] = "tf.ReadVariableOp"([[ACCUM_HANDLE]])
    // CHECK: [[ACCUM_MOMENTUM:%.*]] = "tf.Mul"([[ACCUM]], [[MOMENTUM]])
    // CHECK: [[ACCUM_NEW:%.*]] = "tf.AddV2"([[ACCUM_MOMENTUM]], [[GRAD]])
    // CHECK: "tf.AssignVariableOp"([[ACCUM_HANDLE]], [[ACCUM_NEW]])
    // CHECK: [[ACCUM_NEW_LR:%.*]] = "tf.Mul"([[ACCUM_NEW]], [[LR]])
    // CHECK: [[VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]])
    // CHECK: [[VAR_NEW:%.*]] = "tf.Sub"([[VAR]], [[ACCUM_NEW_LR]])
    // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[VAR_NEW]])
    "tf.ResourceApplyMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyMomentum (with Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_momentum_nesterov
// CHECK-SAME:  [[LR:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>, [[MOMENTUM:%.*]]: tensor<f32>
func.func @decompose_resource_apply_momentum_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"
    // CHECK: [[ACCUM_HANDLE:%.*]] = "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<f32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<f32>>>

    // CHECK: [[ACCUM:%.*]] = "tf.ReadVariableOp"([[ACCUM_HANDLE]])
    // CHECK: [[ACCUM_MOMENTUM:%.*]] = "tf.Mul"([[ACCUM]], [[MOMENTUM]])
    // CHECK: [[ACCUM_NEW:%.*]] = "tf.AddV2"([[ACCUM_MOMENTUM]], [[GRAD]])
    // CHECK: "tf.AssignVariableOp"([[ACCUM_HANDLE]], [[ACCUM_NEW]])
    // CHECK: [[GRAD_LR:%.*]] = "tf.Mul"([[GRAD]], [[LR]])
    // CHECK: [[MOMENTUM_LR:%.*]] = "tf.Mul"([[MOMENTUM]], [[LR]])
    // CHECK: [[ACCUM_NEW_MOMENTUM_LR:%.*]] = "tf.Mul"([[ACCUM_NEW]], [[MOMENTUM_LR]])
    // CHECK: [[DELTA:%.*]] = "tf.AddV2"([[GRAD_LR]], [[ACCUM_NEW_MOMENTUM_LR]])
    // CHECK: [[VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]])
    // CHECK: [[VAR_NEW:%.*]] = "tf.Sub"([[VAR]], [[DELTA]])
    // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[VAR_NEW]])
    "tf.ResourceApplyMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = true} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// -----

// Tests that composite tf.ResourceApplyFtrl{,V2} is decomposed.

// CHECK-LABEL: testResourceApplyFtrl
func.func @testResourceApplyFtrl(%var: tensor<*x!tf_type.resource>, %accum: tensor<*x!tf_type.resource>, %linear: tensor<*x!tf_type.resource>, %grad: tensor<*xf32>, %lr: tensor<*xf32>, %l1: tensor<*xf32>, %l2: tensor<*xf32>, %lr_power: tensor<*xf32>) -> () {
  "tf.ResourceApplyFtrl"(%var, %accum, %linear, %grad, %lr, %l1, %l2, %lr_power) {_tpu_replicate = "cluster_train_function", device = "", multiply_linear_by_lr = false, use_locking = true} : (tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> ()
  // ALWAYS-DECOMPOSE-NOT: ResourceApplyFtrl
  // CHECK: return
  func.return
}

// CHECK-LABEL: testResourceApplyFtrlV2
func.func @testResourceApplyFtrlV2(%var: tensor<*x!tf_type.resource>, %accum: tensor<*x!tf_type.resource>, %linear: tensor<*x!tf_type.resource>, %grad: tensor<*xf32>, %lr: tensor<*xf32>, %l1: tensor<*xf32>, %l2: tensor<*xf32>, %lr_power: tensor<*xf32>) -> () {
  %0 = "tf.ZerosLike"(%lr_power) : (tensor<*xf32>) -> tensor<*xf32>
  "tf.ResourceApplyFtrlV2"(%var, %accum, %linear, %grad, %lr, %l1, %l2, %0, %lr_power) {_tpu_replicate = "cluster_train_function", device = "", multiply_linear_by_lr = false, use_locking = true} : (tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> ()
  // ALWAYS-DECOMPOSE-NOT: ResourceApplyFtrlV2
  // CHECK: return
  func.return
}


// Tests that composite tf.ResourceApplyKerasMomentum (non-Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_keras_momentum_non_nesterov
// CHECK-SAME: (%[[LR:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>, %[[MOMENTUM:.*]]: tensor<f32>)
func.func @decompose_resource_apply_keras_momentum_non_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: %[[VAR_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    // CHECK: %[[ACCUM_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    // CHECK: %[[ACCUM:[0-9]*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: %[[ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[ACCUM]], %[[MOMENTUM]])
    // CHECK: %[[GRAD_LR:[0-9]*]] = "tf.Mul"(%[[GRAD]], %[[LR]])
    // CHECK: %[[NEW_ACCUM:[0-9]*]] = "tf.Sub"(%[[ACCUM_MOMENTUM]], %[[GRAD_LR]])
    // CHECK: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[NEW_ACCUM]])

    // CHECK: %[[VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]])
    // CHECK: %[[NEW_VAR:[0-9]*]] = "tf.AddV2"(%[[VAR]], %[[NEW_ACCUM]])
    // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[NEW_VAR]])

    "tf.ResourceApplyKerasMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyKerasMomentum (with Nesterov) operation
// is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_keras_momentum_nesterov
// CHECK-SAME: (%[[LR:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>, %[[MOMENTUM:.*]]: tensor<f32>)
func.func @decompose_resource_apply_keras_momentum_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: %[[VAR_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    // CHECK: %[[ACCUM_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    // CHECK: %[[ACCUM:[0-9]*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: %[[ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[ACCUM]], %[[MOMENTUM]])
    // CHECK: %[[GRAD_LR:[0-9]*]] = "tf.Mul"(%[[GRAD]], %[[LR]])
    // CHECK: %[[NEW_ACCUM:[0-9]*]] = "tf.Sub"(%[[ACCUM_MOMENTUM]], %[[GRAD_LR]])
    // CHECK: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[NEW_ACCUM]])

    // CHECK: %[[NEW_ACCUM_MOMENTUM:[0-9]*]] = "tf.Mul"(%[[NEW_ACCUM]], %[[MOMENTUM]])
    // CHECK: %[[NEW_DELTA:[0-9]*]] = "tf.Sub"(%[[NEW_ACCUM_MOMENTUM]], %[[GRAD_LR]])
    // CHECK: %[[VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]])
    // CHECK: %[[NEW_VAR:[0-9]*]] = "tf.AddV2"(%[[VAR]], %[[NEW_DELTA]])
    // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[NEW_VAR]])

    "tf.ResourceApplyKerasMomentum"(%0, %1, %arg0, %arg1, %arg2) {use_locking = false, use_nesterov = true} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}



// Tests that composite tf.ResourceApplyAdagradV2 operation is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_adagradv2
// CHECK-SAME: ([[LR:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>)
func.func @decompose_resource_apply_adagradv2(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"()
    // CHECK: [[ACC_HANDLE:%.*]] = "tf.VarHandleOp"()
    // CHECK: [[OLD_ACC:%.*]] = "tf.ReadVariableOp"([[ACC_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: [[GRAD_SQUARE:%.*]] = "tf.Mul"([[GRAD]], [[GRAD]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: [[NEW_ACC:%.*]] = "tf.AddV2"([[OLD_ACC]], [[GRAD_SQUARE]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: [[LR_MULTIPLY:%.*]] = "tf.Mul"([[LR]], [[GRAD]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: [[SQRT:%.*]] = "tf.Sqrt"([[NEW_ACC]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[DIVISOR:%.*]] = "tf.AddV2"([[SQRT]], [[EPSILON]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: [[VAR_DELTA:%.*]] = "tf.Div"([[LR_MULTIPLY]], [[DIVISOR]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[OLD_VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: [[NEW_VAR:%.*]] = "tf.Sub"(%9, %8) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[NEW_VAR]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()
    // CHECK: "tf.AssignVariableOp"([[ACC_HANDLE]], [[NEW_ACC]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    "tf.ResourceApplyAdagradV2"(%0, %1, %arg0, %arg1, %arg2) {update_slots = true, use_locking = true} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}

// CHECK-LABEL: func @decompose_resource_apply_adagrad
// CHECK-SAME:  (%[[LR:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>)
func.func @decompose_resource_apply_adagrad(%arg0: tensor<f32>, %arg1: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    // CHECK: %[[ACCUM_HANDLE:.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    // CHECK: %[[ACCUM:.*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: %[[GRAD_SQUARE:.*]] = "tf.Mul"(%[[GRAD]], %[[GRAD]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[ACCUM_NEW:.*]] = "tf.AddV2"(%[[ACCUM]], %[[GRAD_SQUARE]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: %[[LR_MULTIPLY:.*]] = "tf.Mul"(%[[LR]], %[[GRAD]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[SQRT:.*]] = "tf.Sqrt"(%[[ACCUM_NEW]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[DIV:.*]] = "tf.Div"(%[[LR_MULTIPLY]], %[[SQRT]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[VAR:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: %[[VAR_NEW:.*]] = "tf.Sub"(%[[VAR]], %[[DIV]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[VAR_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()
    // CHECK: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[ACCUM_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    "tf.ResourceApplyAdagrad"(%0, %1, %arg0, %arg1) {update_slots = true, use_locking = true} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyAdam (non-Nesterov) operation is
// decomposed.

// CHECK-LABEL: func @decompose_resource_apply_adam_non_nesterov
// CHECK-SAME: ([[BETA1_POWER:%.*]]: tensor<f32>, [[BETA2_POWER:%.*]]: tensor<f32>, [[LR:%.*]]: tensor<f32>, [[BETA1:%.*]]: tensor<f32>, [[BETA2:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>)
func.func @decompose_resource_apply_adam_non_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>) -> () {
  "tf_device.cluster"() ({

    // CHECK: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"()
    // CHECK: [[M_HANDLE:%.*]] = "tf.VarHandleOp"()
    // CHECK: [[V_HANDLE:%.*]] = "tf.VarHandleOp"()
    // CHECK: [[ONE_MINUS_BETA2_POWER:%.*]] = "tf.Sub"([[ONE]], [[BETA2_POWER]])
    // CHECK: [[SQRT_ONE_MINUS_BETA2_POWER:%.*]] = "tf.Sqrt"([[ONE_MINUS_BETA2_POWER]])
    // CHECK: [[ONE_MINUS_BETA1_POWER:%.*]] = "tf.Sub"([[ONE]], [[BETA1_POWER]])
    // CHECK: [[ALPHA_NO_LR:%.*]] = "tf.Div"([[SQRT_ONE_MINUS_BETA2_POWER]], [[ONE_MINUS_BETA1_POWER]])
    // CHECK: [[ALPHA:%.*]] = "tf.Mul"([[LR]], [[ALPHA_NO_LR]])
    // CHECK: [[OLD_M:%.*]] = "tf.ReadVariableOp"([[M_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: [[BETA1_OLD_M:%.*]] = "tf.Mul"([[BETA1]], [[OLD_M]])
    // CHECK: [[ONE_MINUS_BETA1:%.*]] = "tf.Sub"([[ONE]], [[BETA1]])
    // CHECK: [[ONE_MINUS_BETA1_GRAD:%.*]] = "tf.Mul"([[ONE_MINUS_BETA1]], [[GRAD]])
    // CHECK: [[NEW_M:%.*]] = "tf.AddV2"([[BETA1_OLD_M]], [[ONE_MINUS_BETA1_GRAD]])
    // CHECK: [[OLD_V:%.*]] = "tf.ReadVariableOp"([[V_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: [[BETA2_OLD_V:%.*]] = "tf.Mul"([[BETA2]], [[OLD_V]])
    // CHECK: [[ONE_MINUS_BETA2:%.*]] = "tf.Sub"([[ONE]], [[BETA2]])
    // CHECK: [[GRAD_SQUARE:%.*]] = "tf.Square"([[GRAD]])
    // CHECK: [[V_DELTA:%.*]] = "tf.Mul"([[ONE_MINUS_BETA2]], [[GRAD_SQUARE]])
    // CHECK: [[NEW_V:%.*]] = "tf.AddV2"([[BETA2_OLD_V]], [[V_DELTA]])
    // CHECK: [[ALPHA_NEW_M:%.*]] = "tf.Mul"([[ALPHA]], [[NEW_M]])
    // CHECK: [[SQRT_NEW_V:%.*]] = "tf.Sqrt"([[NEW_V]])
    // CHECK: [[SQRT_NEW_V_EPSILON:%.*]] = "tf.AddV2"([[SQRT_NEW_V]], [[EPSILON]])
    // CHECK: [[VAR_DELTA:%.*]] = "tf.Div"([[ALPHA_NEW_M]], [[SQRT_NEW_V_EPSILON]])
    // CHECK: [[OLD_VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: [[NEW_VAR:%.*]] = "tf.Sub"([[OLD_VAR]], [[VAR_DELTA]])
    // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[NEW_VAR]])
    // CHECK: "tf.AssignVariableOp"([[M_HANDLE]], [[NEW_M]])
    // CHECK: "tf.AssignVariableOp"([[V_HANDLE]], [[NEW_V]])

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    "tf.ResourceApplyAdam"(%0, %1, %2, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {use_locking = false, use_nesterov = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceApplyAdam (with Nesterov) operation is
// decomposed.

// CHECK-LABEL: func @decompose_resource_apply_adam_nesterov(
// CHECK-SAME:  [[BETA1_POWER:%.*]]: tensor<f32>, [[BETA2_POWER:%.*]]: tensor<f32>, [[LR:%.*]]: tensor<f32>, [[BETA1:%.*]]: tensor<f32>, [[BETA2:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>) {
func.func @decompose_resource_apply_adam_nesterov(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>) -> () {
  "tf_device.cluster"() ({

  // CHECK: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
  // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
  // CHECK: [[M_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
  // CHECK: [[V_HANDLE:%.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "v"}
  // CHECK: [[VAL_82:%.*]] = "tf.Sub"([[ONE]], [[BETA2_POWER]])
  // CHECK: [[VAL_83:%.*]] = "tf.Sqrt"([[VAL_82]])
  // CHECK: [[VAL_84:%.*]] = "tf.Sub"([[ONE]], [[BETA1_POWER]])
  // CHECK: [[VAL_85:%.*]] = "tf.Div"([[VAL_83]], [[VAL_84]])
  // CHECK: [[VAL_86:%.*]] = "tf.Mul"([[LR]], [[VAL_85]])
  // CHECK: [[OLD_M:%.*]] = "tf.ReadVariableOp"([[M_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
  // CHECK: [[VAL_88:%.*]] = "tf.Mul"([[BETA1]], [[OLD_M]])
  // CHECK: [[VAL_89:%.*]] = "tf.Sub"([[ONE]], [[BETA1]])
  // CHECK: [[VAL_90:%.*]] = "tf.Mul"([[VAL_89]], [[GRAD]])
  // CHECK: [[NEW_M:%.*]] = "tf.AddV2"([[VAL_88]], [[VAL_90]])
  // CHECK: [[OLD_V:%.*]] = "tf.ReadVariableOp"([[V_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
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
  // CHECK: [[OLD_VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
  // CHECK: [[NEW_VAR:%.*]] = "tf.Sub"([[OLD_VAR]], [[VAL_105]])
  // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[NEW_VAR]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()
  // CHECK: "tf.AssignVariableOp"([[M_HANDLE]], [[NEW_M]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()
  // CHECK: "tf.AssignVariableOp"([[V_HANDLE]], [[NEW_V]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*xf32>) -> ()

    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>
    %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xf32>>>

    "tf.ResourceApplyAdam"(%0, %1, %2, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {use_locking = false, use_nesterov = true} : (tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<*x!tf_type.resource<tensor<*xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}

// Tests that composite tf.ResourceApplyAdam operation with complex types is
// decomposed.

// CHECK-LABEL: @decompose_adam_with_complex_inputs
func.func @decompose_adam_with_complex_inputs(%arg0: tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, %arg1: tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, %arg2: tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, %arg3: tensor<complex<f32>>, %arg4: tensor<complex<f32>>, %arg5: tensor<complex<f32>>, %arg6: tensor<complex<f32>>, %arg7: tensor<complex<f32>>, %arg8: tensor<complex<f32>>, %arg9: tensor<2xcomplex<f32>>) attributes {tf.entry_function = {control_outputs = "Adam/update_Variable_1/ResourceApplyAdam", inputs = "_arg0,_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,_arg8,_arg9", outputs = ""}} {
  "tf_device.cluster"() ({

    // CHECK: "tf.Const"() {value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>} : () -> tensor<complex<f32>>
    // CHECK-NOT: tf.ResourceApplyAdam
    "tf.ResourceApplyAdam"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) {_XlaHasReferenceVars = false, _xla_inferred_shapes = [], device = "/job:localhost/replica:0/task:0/device:TPU:0", use_locking = false, use_nesterov = false} : (tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, tensor<!tf_type.resource<tensor<2xcomplex<f32>>>>, tensor<complex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>, tensor<2xcomplex<f32>>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}

// Tests that composite tf.ResourceGather operation is decomposed.

// CHECK-LABEL: @decompose_resource_gather_op
// CHECK-SAME: [[INDEX:%.+]]: tensor<?xi32>
func.func @decompose_resource_gather_op(%indices : tensor<?xi32>) -> tensor<*xi32> {
  %0 = "tf_device.cluster"() ({
    // CHECK: [[ZERO:%.+]] = "tf.Const"() {value = dense<0> : tensor<i64>}

    // CHECK: [[VAR:%.+]] = "tf.VarHandleOp"
    %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>

    // CHECK: [[READVAR:%.+]] = "tf.ReadVariableOp"([[VAR]])
    // CHECK: [[GATHER:%.+]] = "tf.GatherV2"([[READVAR]], [[INDEX]], [[ZERO]]) {batch_dims = 0 : i64} : (tensor<*xi32>, tensor<?xi32>, tensor<i64>) -> tensor<*xi32>
    // CHECK: return [[GATHER]]
    %1 = "tf.ResourceGather"(%resource, %indices) : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<?xi32>) -> (tensor<*xi32>)
    tf_device.return %1 : tensor<*xi32>
  }) : () -> (tensor<*xi32>)
  func.return %0: tensor<*xi32>
}



// Tests that resource subtype is correctly propagated when decomposing tf.ResourceGather.

// CHECK-LABEL: @decompose_resource_gather_op_subtype
func.func @decompose_resource_gather_op_subtype(%indices : tensor<5xi32>) -> tensor<2x5x16xi32> {
  %0 = "tf_device.cluster"() ({
    %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8x16xi32>>>

    // CHECK: "tf.GatherV2"({{.+}}, {{.+}}, {{.+}}) {batch_dims = 1 : i64} : (tensor<2x8x16xi32>, tensor<5xi32>, tensor<i64>) -> tensor<2x5x16xi32>
    %1 = "tf.ResourceGather"(%resource, %indices) {batch_dims = 1} : (tensor<*x!tf_type.resource<tensor<2x8x16xi32>>>, tensor<5xi32>) -> (tensor<2x5x16xi32>)

    tf_device.return %1 : tensor<2x5x16xi32>
  }) : () -> (tensor<2x5x16xi32>)
  func.return %0: tensor<2x5x16xi32>
}


// Tests that composite tf.ResourceApplyCenteredRMSProp operation is decomposed.

// CHECK-LABEL: func @decompose_resource_apply_centered_RMS_prop
// CHECK-SAME:  [[VAR:%.*]]: tensor<f32>, [[MG:%.*]]: tensor<f32>, [[MS:%.*]]: tensor<f32>, [[MOM:%.*]]: tensor<f32>, [[LR:%.*]]: tensor<f32>, [[RHO:%.*]]: tensor<f32>, [[MOMENTUM:%.*]]: tensor<f32>, [[EPSILON:%.*]]: tensor<f32>, [[GRAD:%.*]]: tensor<f32>
func.func @decompose_resource_apply_centered_RMS_prop(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>) -> () {
  "tf_device.cluster"() ({
    // CHECK: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    // CHECK: [[VAR_HANDLE:%.*]] = "tf.VarHandleOp"
    // CHECK: [[MG_HANDLE:%.*]] = "tf.VarHandleOp"
    // CHECK: [[MS_HANDLE:%.*]] = "tf.VarHandleOp"
    // CHECK: [[MOM_HANDLE:%.*]] = "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<f32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<f32>>>
    %2 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<f32>>>
    %3 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<f32>>>

    // CHECK: [[GRADSQ:%.*]] = "tf.Mul"([[GRAD]], [[GRAD]])
    // CHECK: [[SB:%.*]] = "tf.Sub"([[ONE]], [[RHO]])
    // CHECK: [[GRAD_SUB:%.*]] = "tf.Mul"([[GRADSQ]], [[SB]])
    // CHECK: [[MS:%.*]] = "tf.ReadVariableOp"([[MS_HANDLE]])
    // CHECK: [[MS_RHO:%.*]] = "tf.Mul"([[MS]], [[RHO]])
    // CHECK: [[MS_NEW:%.*]] = "tf.AddV2"([[GRAD_SUB]], [[MS_RHO]])
    // CHECK: "tf.AssignVariableOp"([[MS_HANDLE]], [[MS_NEW]])

    // CHECK: [[SUB_RHO:%.*]] = "tf.Sub"([[ONE]], [[RHO]])
    // CHECK: [[SUB_GRAD:%.*]] = "tf.Mul"([[GRAD]], [[SUB_RHO]])
    // CHECK: [[MG:%.*]] = "tf.ReadVariableOp"([[MG_HANDLE]])
    // CHECK: [[MG_RHO:%.*]] = "tf.Mul"([[MG]], [[RHO]])
    // CHECK: [[MG_NEW:%.*]] = "tf.AddV2"([[SUB_GRAD]], [[MG_RHO]])
    // CHECK: "tf.AssignVariableOp"([[MG_HANDLE]], [[MG_NEW]])

    // CHECK: [[MOM:%.*]] = "tf.ReadVariableOp"([[MOM_HANDLE]])
    // CHECK: [[MOM_MOM:%.*]] = "tf.Mul"([[MOMENTUM]], [[MOM]])
    // CHECK: [[LR_GRAD:%.*]] = "tf.Mul"([[LR]], [[GRAD]])

    // CHECK: [[MG_MG:%.*]] = "tf.Square"([[MG_NEW]])
    // CHECK: [[MG_SUB:%.*]] = "tf.Sub"([[MS_NEW]], [[MG_MG]])
    // CHECK: [[MG_NEW:%.*]] = "tf.AddV2"([[MG_SUB]], [[EPSILON]])
    // CHECK: [[MG_SQRT:%.*]] = "tf.Sqrt"([[MG_NEW]])
    // CHECK: [[MOM_DIV:%.*]] = "tf.Div"([[LR_GRAD]], [[MG_SQRT]])
    // CHECK: [[MOM_NEW:%.*]] = "tf.AddV2"([[MOM_MOM]], [[MOM_DIV]])

    // CHECK: [[VAR:%.*]] = "tf.ReadVariableOp"([[VAR_HANDLE]])
    // CHECK: [[VAR_NEW:%.*]] = "tf.Sub"([[VAR]], [[MOM_NEW]])
    // CHECK: "tf.AssignVariableOp"([[VAR_HANDLE]], [[VAR_NEW]])

    "tf.ResourceApplyCenteredRMSProp"(%0, %1, %2, %3, %arg4, %arg5, %arg6, %arg7, %arg8) {use_locking = false} : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// CHECK-LABEL: func @decompose_resource_apply_RMS_prop
// CHECK-SAME:  (%[[VAR_HANDLE:.*]]: tensor<*x!tf_type.resource>, %[[MS_HANDLE:.*]]: tensor<*x!tf_type.resource>, %[[MOM_HANDLE:.*]]: tensor<*x!tf_type.resource>,
// CHECK-SAME:   %[[LR:.*]]: tensor<f32>, %[[RHO:.*]]: tensor<f32>, %[[MOMENTUM:.*]]: tensor<f32>, %[[EPSILON:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<f32>)
func.func @decompose_resource_apply_RMS_prop(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>) -> () {
  "tf_device.cluster"() ({
    // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %[[MS:.*]] = "tf.ReadVariableOp"(%[[MS_HANDLE]]) : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    // CHECK: %[[MS_RHO:.*]] = "tf.Mul"(%[[MS]], %[[RHO]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: %[[GRAD_SQUARE:.*]] = "tf.Square"(%[[GRAD]]) : (tensor<f32>) -> tensor<f32>
    // CHECK: %[[ONE_RHO:.*]] = "tf.Sub"(%[[ONE]], %[[RHO]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[GRAD_SQUARE]], %[[ONE_RHO]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[MS_NEW:.*]] = "tf.AddV2"(%[[MS_RHO]], %[[MUL]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: "tf.AssignVariableOp"(%[[MS_HANDLE]], %[[MS_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> ()
    // CHECK: %[[MOM:.*]] = "tf.ReadVariableOp"(%[[MOM_HANDLE]]) : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    // CHECK: %[[MOMENTUM_MOM:.*]] = "tf.Mul"(%[[MOMENTUM]], %[[MOM]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[LR_GRAD:.*]] = "tf.Mul"(%[[LR]], %[[GRAD]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[MS_NEW]], %[[EPSILON]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: %[[SQRT:.*]] = "tf.Sqrt"(%[[ADD]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[DIV:.*]] = "tf.Div"(%[[LR_GRAD]], %[[SQRT]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[MOM_NEW:.*]] = "tf.AddV2"(%[[MOMENTUM_MOM]], %[[DIV]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: "tf.AssignVariableOp"(%[[MOM_HANDLE]], %[[MOM_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> ()
    // CHECK: %[[VAR:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]]) : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    // CHECK: %[[VAR_NEW:.*]] = "tf.Sub"(%[[VAR]], %[[MOM_NEW]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[VAR_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> ()
    "tf.ResourceApplyRMSProp"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) {use_locking = false} : (tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceScatterAdd operation is decomposed.

// CHECK-LABEL: @decompose_resource_scatter_add_op
// CHECK-SAME: ([[INDEX:%.+]]: tensor<2x?xi32>, [[UPDATE:%.+]]: tensor<?x?x?xi32>)
func.func @decompose_resource_scatter_add_op(%indices : tensor<2x?xi32>, %updates: tensor<?x?x?xi32>) {
  // CHECK: [[CST:%.+]] = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  "tf_device.cluster"() ({
    // CHECK: [[VAR:%.+]] = "tf.VarHandleOp"
    %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>

    // CHECK: [[READ:%.+]] = "tf.ReadVariableOp"([[VAR]])
    // CHECK: [[EXPAND:%.+]] = "tf.ExpandDims"([[INDEX]], [[CST]]) : (tensor<2x?xi32>, tensor<i32>) -> tensor<2x?x1xi32>
    // CHECK: [[TENSOR:%.+]] = "tf.TensorScatterAdd"([[READ]], [[EXPAND]], [[UPDATE]]) : (tensor<*xi32>, tensor<2x?x1xi32>, tensor<?x?x?xi32>) -> tensor<*xi32>
    // CHECK: "tf.AssignVariableOp"([[VAR]], [[TENSOR]])
    "tf.ResourceScatterAdd"(%resource, %indices, %updates) : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<2x?xi32>, tensor<?x?x?xi32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}

// Tests that composite tf.ResourceScatterAdd with 1d Indices operation is decomposed.

// CHECK-LABEL: @decompose_resource_scatter_add_op_1d_indices
// CHECK-SAME: ([[INDEX:%.+]]: tensor<?xi32>, [[UPDATE:%.+]]: tensor<?x?x?xi32>)
func.func @decompose_resource_scatter_add_op_1d_indices(%indices : tensor<?xi32>, %updates: tensor<?x?x?xi32>) {
  // CHECK: [[CST:%.+]] = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  "tf_device.cluster"() ({
    // CHECK: [[VAR:%.+]] = "tf.VarHandleOp"
    %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>

    // CHECK: [[READ:%.+]] = "tf.ReadVariableOp"([[VAR]])
    // CHECK: [[EXPAND:%.+]] = "tf.ExpandDims"([[INDEX]], [[CST]]) : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
    // CHECK: [[TENSOR:%.+]] = "tf.TensorScatterAdd"([[READ]], [[EXPAND]], [[UPDATE]]) : (tensor<*xi32>, tensor<?x1xi32>, tensor<?x?x?xi32>) -> tensor<*xi32>
    // CHECK: "tf.AssignVariableOp"([[VAR]], [[TENSOR]])
    "tf.ResourceScatterAdd"(%resource, %indices, %updates) : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<?xi32>, tensor<?x?x?xi32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Tests that composite tf.ResourceScatterUpdate operation is decomposed.

// CHECK-LABEL: @decompose_resource_scatter_update_op
// CHECK-SAME: ([[INDEX:%.+]]: tensor<2x?xi32>, [[UPDATE:%.+]]: tensor<?x?x?xi32>)
func.func @decompose_resource_scatter_update_op(%indices : tensor<2x?xi32>, %updates: tensor<?x?x?xi32>) {
  // CHECK: [[CST:%.+]] = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  "tf_device.cluster"() ({
    // CHECK: [[VAR:%.+]] = "tf.VarHandleOp"
    %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>

    // CHECK: [[READ:%.+]] = "tf.ReadVariableOp"([[VAR]])
    // CHECK: [[EXPAND:%.+]] = "tf.ExpandDims"([[INDEX]], [[CST]]) : (tensor<2x?xi32>, tensor<i32>) -> tensor<2x?x1xi32>
    // CHECK: [[TENSOR:%.+]] = "tf.TensorScatterUpdate"([[READ]], [[EXPAND]], [[UPDATE]]) : (tensor<*xi32>, tensor<2x?x1xi32>, tensor<?x?x?xi32>) -> tensor<*xi32>
    // CHECK: "tf.AssignVariableOp"([[VAR]], [[TENSOR]])
    "tf.ResourceScatterUpdate"(%resource, %indices, %updates) : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<2x?xi32>, tensor<?x?x?xi32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}

// Test that tf.ResourceScatterUpdate operation is decomposed to
// tf.TensorScatterUpdate. updates is scalar
// CHECK-LABEL: @decompose_resource_scatter_update_op_with_scalar_updates
func.func @decompose_resource_scatter_update_op_with_scalar_updates(%resource : tensor<*x!tf_type.resource>, %indices : tensor<?xi32>, %updates: tensor<i32>) {
  "tf_device.cluster"() ({
    // CHECK-NOT: ResourceScatterUpdate
    // CHECK-NOT: BroadcastTo
    // CHECK: TensorScatterUpdate
    "tf.ResourceScatterUpdate"(%resource, %indices, %updates) {device = ""} : (tensor<*x!tf_type.resource>, tensor<?xi32>, tensor<i32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// Test that composite tf.ResourceScatterUpdate operation is decomposed to tf.TensorScatterUpdate when update is unranked
// CHECK-LABEL: @decompose_unranked_updates
func.func @decompose_unranked_updates(%resource : tensor<*x!tf_type.resource>, %indices : tensor<?xi32>, %updates: tensor<*xi32>) {
  "tf_device.cluster"() ({
    // CHECK-NOT: ResourceScatterUpdate
    // CHECK-NOT: BroadcastTo
    // CHECK: TensorScatterUpdate
    "tf.ResourceScatterUpdate"(%resource, %indices, %updates) {device = ""} : (tensor<*x!tf_type.resource>, tensor<?xi32>, tensor<*xi32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// CHECK-LABEL: @decompose_scalar_updates
func.func @decompose_scalar_updates(%arg0: tensor<!tf_type.resource<tensor<4x3xi32>>>, %arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  "tf_device.cluster"() ({
    // CHECK-NOT: ResourceScatterUpdate
    // CHECK-NOT: BroadcastTo
    // CHECK: TensorScatterUpdate
    "tf.ResourceScatterUpdate"(%arg0, %arg1, %arg2) : (tensor<!tf_type.resource<tensor<4x3xi32>>>, tensor<2xi32>, tensor<i32>) -> ()
    tf_device.return
  }) : () -> ()
  func.return
}

// Tests that tf.VariableShape operation is decomposed.

// CHECK-LABEL: @decompose_variable_shape_i32
func.func @decompose_variable_shape_i32(%input: tensor<!tf_type.resource<tensor<?x?x?xf32>>>) -> tensor<3xi32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.VariableShape"(%input) : (tensor<!tf_type.resource<tensor<?x?x?xf32>>>) -> tensor<3xi32>
    // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%arg0)
    // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%[[READ]])
    // CHECK: return %[[SHAPE]]
    tf_device.return %1 : tensor<3xi32>
  }) : () -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @decompose_variable_shape_i64
func.func @decompose_variable_shape_i64(%input: tensor<!tf_type.resource<tensor<?x?x?xf32>>>) -> tensor<3xi64> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.VariableShape"(%input) : (tensor<!tf_type.resource<tensor<?x?x?xf32>>>) -> tensor<3xi64>
    // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%arg0)
    // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%[[READ]])
    // CHECK: return %[[SHAPE]]
    tf_device.return %1 : tensor<3xi64>
  }) : () -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// CHECK-LABEL: @decompose_variable_shape_no_subtype
func.func @decompose_variable_shape_no_subtype(%input: tensor<!tf_type.resource>) -> tensor<3xi32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.VariableShape"(%input) : (tensor<!tf_type.resource>) -> tensor<3xi32>
    // CHECK: "tf.VariableShape"
    // CHECK-NOT: "tf.ReadVariableOp"
    // CHECK-NOT: "tf.Shape"
    tf_device.return %1 : tensor<3xi32>
  }) : () -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}


// Tests that resource subtype is correctly propagated when decomposing tf.ResourceGather.

// CHECK-LABEL: @decompose_resource_apply_proximal_adagrad_op
// CHECK-SAME: (%[[LR:.*]]: tensor<f32>, %[[L1:.*]]: tensor<f32>, %[[L2:.*]]: tensor<f32>, %[[GRAD:.*]]: tensor<4xf32>)
func.func @decompose_resource_apply_proximal_adagrad_op(%lr: tensor<f32>, %l1: tensor<f32>, %l2: tensor<f32>, %grad: tensor<4xf32>) -> () {
  "tf_device.cluster"() ({
    %var = "tf.VarHandleOp"() {container = "c", shared_name = "var"} : () -> tensor<*x!tf_type.resource<tensor<4xf32>>>
    %accum = "tf.VarHandleOp"() {container = "c", shared_name = "accum"} : () -> tensor<*x!tf_type.resource<tensor<4xf32>>>

    // CHECK-DAG: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "var"} : () -> tensor<*x!tf_type.resource<tensor<4xf32>>>
    // CHECK-DAG: %[[ACCUM_HANDLE:.*]] = "tf.VarHandleOp"() {container = "c", shared_name = "accum"} : () -> tensor<*x!tf_type.resource<tensor<4xf32>>>
    // CHECK-DAG: %[[GRAD_SQ:.*]] = "tf.Square"(%[[GRAD]]) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[ACCUM:.*]] = "tf.ReadVariableOp"(%[[ACCUM_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
    // CHECK-DAG: %[[ACCUM_NEW:.*]] = "tf.AddV2"(%[[ACCUM]], %[[GRAD_SQ]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[RSQRT_ACCUM:.*]] = "tf.Rsqrt"(%[[ACCUM_NEW]]) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[ADAGRAD_LR:.*]] = "tf.Mul"(%[[LR]], %[[RSQRT_ACCUM]]) : (tensor<f32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[DELTA:.*]] = "tf.Mul"(%[[GRAD]], %[[ADAGRAD_LR]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[VAR:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]]) : (tensor<*x!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
    // CHECK-DAG: %[[PROX:.*]] = "tf.Sub"(%[[VAR]], %[[DELTA]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[SIGN:.*]] = "tf.Sign"(%[[PROX]]) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[ABS:.*]] = "tf.Abs"(%[[PROX]]) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[SCALED_L1:.*]] = "tf.Mul"(%[[ADAGRAD_LR]], %[[L1]]) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK-DAG: %[[PROX_NEW:.*]] = "tf.Sub"(%[[ABS]], %[[SCALED_L1]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[MAX:.*]] = "tf.Maximum"(%[[PROX_NEW]], %[[ZERO]]) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK-DAG: %[[SIGNED:.*]] = "tf.Mul"(%[[SIGN]], %[[MAX]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[GT:.*]] = "tf.Greater"(%[[L1]], %[[ZERO]]) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    // CHECK-DAG: %[[NUMERATOR:.*]] = "tf.SelectV2"(%[[GT]], %[[SIGNED:.*]], %[[PROX]]) : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[SCALED_L2:.*]] = "tf.Mul"(%[[ADAGRAD_LR]], %[[L2]]) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK-DAG: %[[DENOMINATOR:.*]] = "tf.Add"(%[[ONE]], %[[SCALED_L2]]) : (tensor<f32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: %[[VAR_NEW:.*]] = "tf.Div"(%[[NUMERATOR]], %[[DENOMINATOR]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[VAR_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<4xf32>) -> ()
    // CHECK-DAG: "tf.AssignVariableOp"(%[[ACCUM_HANDLE]], %[[ACCUM_NEW]]) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<4xf32>) -> ()

    "tf.ResourceApplyProximalAdagrad"(%var, %accum, %lr, %l1, %l2, %grad) {use_locking = false} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>) -> ()

    tf_device.return
  }) : () -> ()
  func.return
}


// Test that tf.RngReadAndSkip op is decomposed.
// CHECK-LABEL: func @decompose_rng_read_and_skip_op
func.func @decompose_rng_read_and_skip_op(%resource: tensor<!tf_type.resource<tensor<3xi64>>>) -> tensor<3xi64> {
  %0 = "tf_device.cluster"() ({
    // We rely on the TensorFlow StatefulRandomOpsTest to check it is lowered
    // correctly.
    // CHECK-NOT: tf.RngReadAndSkip
    %alg = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %delta = "tf.Const"() {value = dense<10> : tensor<ui64>} : () -> tensor<ui64>
    %1 = "tf.RngReadAndSkip"(%resource, %alg, %delta) : (tensor<!tf_type.resource<tensor<3xi64>>>, tensor<i32>, tensor<ui64>) -> tensor<3xi64>
    tf_device.return %1 : tensor<3xi64>
  }) : () -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// Test that ordering token resource of tf.CollectiveReduceV2 op is removed.
// CHECK-LABEL: func @decompose_collective_reduce_v2_op
// CHECK-SAME: [[TOKEN:%.+]]: tensor<!tf_type.resource
func.func @decompose_collective_reduce_v2_op(%input: tensor<3xi64>,
  %group_size: tensor<i32>,
  %group_key: tensor<i32>,
  %instance_key: tensor<i32>,
  %token: tensor<!tf_type.resource<tensor<f32>>>
) -> tensor<3xi64> {
  %0 = "tf_device.cluster"() ({
    // arg4 shall have been removed.
    // CHECK: tf.CollectiveReduceV2
    // CHECK-NOT: [[TOKEN]]
    // CHECK-SAME: merge_op
    %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key, %token) {
                merge_op = "Add", final_op = "Id" } : (
            tensor<3xi64>, tensor<i32>, tensor<i32>, tensor<i32>,
            tensor<!tf_type.resource<tensor<f32>>>) -> tensor<3xi64>
    tf_device.return %0 : tensor<3xi64>
  }) : () -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}
