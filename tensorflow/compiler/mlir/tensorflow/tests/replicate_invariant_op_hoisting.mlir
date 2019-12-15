// RUN: tf-opt %s -tf-replicate-invariant-op-hoisting | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @replicate_arg_shape
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xf32>)
func @replicate_arg_shape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0:4 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*xf32>) {n = 2: i32} {
    %1 = "tf.Shape"(%ri) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %2 = "tf.opA"(%1, %ri) : (tensor<?xi32>, tensor<*xf32>) -> tensor<*xi32>
    tf_device.return %1, %2 : tensor<?xi32>, tensor<*xi32>
  }
  return
}

// CHECK: %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_0]])
// CHECK: tf_device.replicate([%[[ARG_0]], %[[ARG_1]]] as %[[RI:[a-z0-9]*]]: tensor<*xf32>)
// CHECK:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[SHAPE]], %[[RI]])
// CHECK:   tf_device.return %[[SHAPE]], %[[OP_A]]


// CHECK-LABEL: func @invariant_shape
// CHECK-SAME: (%{{[a-z0-9]*}}: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xf32>)
func @invariant_shape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0:2 = tf_device.replicate([%arg0, %arg0] as %ri: tensor<*xf32>) {n = 2: i32} {
    %1 = "tf.Shape"(%arg1) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    tf_device.return %1 : tensor<?xi32>
  }
  return
}

// CHECK: %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_1]])
// CHECK: tf_device.replicate
// CHECK:   tf_device.return %[[SHAPE]]


// CHECK-LABEL: func @replicate_resource_var_arg_shape
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*x!tf.resource>, %[[ARG_1:[a-z0-9]*]]: tensor<*x!tf.resource>)
func @replicate_resource_var_arg_shape(%arg0: tensor<*x!tf.resource>, %arg1: tensor<*x!tf.resource>) {
  %0:6 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*x!tf.resource>) {n = 2: i32} {
    %1 = "tf.ReadVariableOp"(%ri) {dtype = "tfdtype$DT_FLOAT"} : (tensor<*x!tf.resource>) -> tensor<*xf32>
    %2 = "tf.Shape"(%1) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %3 = "tf.opA"(%1, %2, %ri) : (tensor<*xf32>, tensor<?xi32>, tensor<*x!tf.resource>) -> tensor<*xi32>
    tf_device.return %1, %2, %3 : tensor<*xf32>, tensor<?xi32>, tensor<*xi32>
  }
  return
}

// CHECK: %[[VAR_SHAPE:[0-9]*]] = "tf.VariableShape"(%[[ARG_0]])
// CHECK: tf_device.replicate([%[[ARG_0]], %[[ARG_1]]] as %[[RI:[a-z0-9]*]]: tensor<*x!tf.resource>)
// CHECK:   %[[READ_VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[RI]])
// CHECK:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[READ_VAR]], %[[VAR_SHAPE]], %[[RI]])
// CHECK:   tf_device.return %[[READ_VAR]], %[[VAR_SHAPE]], %[[OP_A]]


// CHECK-LABEL: func @invariant_resource_var_shape
// CHECK-SAME: (%{{[a-z0-9]*}}: tensor<*x!tf.resource>, %[[ARG_1:[a-z0-9]*]]: tensor<*x!tf.resource>)
func @invariant_resource_var_shape(%arg0: tensor<*x!tf.resource>, %arg1: tensor<*x!tf.resource>) {
  %0 = "tf.ReadVariableOp"(%arg1) {dtype = "tfdtype$DT_FLOAT"} : (tensor<*x!tf.resource>) -> tensor<*xf32>
  %1:2 = tf_device.replicate([%arg0, %arg0] as %ri: tensor<*x!tf.resource>) {n = 2: i32} {
    %2 = "tf.Shape"(%0) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    tf_device.return %2 : tensor<?xi32>
  }
  return
}

// CHECK: %[[READ_VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[ARG_1]])
// CHECK: %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[READ_VAR]])
// CHECK: tf_device.replicate
// CHECK:   tf_device.return %[[SHAPE]]


// CHECK-LABEL: func @dependent_invariants
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %{{[a-z0-9]*}}: tensor<*xf32>)
func @dependent_invariants(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0:6 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*xf32>) {n = 2: i32} {
    %1 = "tf.Shape"(%ri) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %2 = "tf.opA"(%1) : (tensor<?xi32>) -> tensor<*xi32>
    %3 = "tf.opB"(%1, %2) : (tensor<?xi32>, tensor<*xi32>) -> tensor<*xf32>
    tf_device.return %1, %2, %3 : tensor<?xi32>, tensor<*xi32>, tensor<*xf32>
  }
  return
}

// CHECK: %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_0]])
// CHECK: %[[OP_A:[0-9]*]] = "tf.opA"(%[[SHAPE]])
// CHECK: %[[OP_B:[0-9]*]] = "tf.opB"(%[[SHAPE]], %[[OP_A]])
// CHECK: tf_device.replicate
// CHECK:   tf_device.return %[[SHAPE]], %[[OP_A]], %[[OP_B]]
