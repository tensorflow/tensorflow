// RUN: tf-opt %s -tf-replicate-invariant-op-hoisting | FileCheck %s

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
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*x!tf_type.resource>, %[[ARG_1:[a-z0-9]*]]: tensor<*x!tf_type.resource>)
func @replicate_resource_var_arg_shape(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>) {
  %0:6 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*x!tf_type.resource>) {n = 2: i32} {
    %1 = "tf.ReadVariableOp"(%ri) {dtype = "tfdtype$DT_FLOAT"} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    %2 = "tf.Shape"(%1) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %3 = "tf.opA"(%1, %2, %ri) : (tensor<*xf32>, tensor<?xi32>, tensor<*x!tf_type.resource>) -> tensor<*xi32>
    tf_device.return %1, %2, %3 : tensor<*xf32>, tensor<?xi32>, tensor<*xi32>
  }
  return
}

// CHECK: %[[VAR_SHAPE:[0-9]*]] = "tf.VariableShape"(%[[ARG_0]])
// CHECK: tf_device.replicate([%[[ARG_0]], %[[ARG_1]]] as %[[RI:[a-z0-9]*]]: tensor<*x!tf_type.resource>)
// CHECK:   %[[READ_VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[RI]])
// CHECK:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[READ_VAR]], %[[VAR_SHAPE]], %[[RI]])
// CHECK:   tf_device.return %[[READ_VAR]], %[[VAR_SHAPE]], %[[OP_A]]

// CHECK-LABEL: func @replicate_arg_shape_with_packed
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xf32>)
func @replicate_arg_shape_with_packed(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) {
  %0:4 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*xf32>, %arg2 as %rj : tensor<*xf32>) {n = 2: i32} {
    %1 = "tf.Shape"(%rj) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %2 = "tf.opA"(%1, %rj) : (tensor<?xi32>, tensor<*xf32>) -> tensor<*xi32>
    tf_device.return %1, %2 : tensor<?xi32>, tensor<*xi32>
  }
  return
}

// CHECK: %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_2]])
// CHECK: tf_device.replicate([%[[ARG_0]], %[[ARG_1]]] as %[[RI:[a-z0-9]*]]: tensor<*xf32>, %[[ARG_2]] as %[[RJ:[a-z0-9]*]]: tensor<*xf32>)
// CHECK:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[SHAPE]], %[[RJ]])
// CHECK:   tf_device.return %[[SHAPE]], %[[OP_A]]

// CHECK-LABEL: func @replicate_resource_var_arg_shape_with_packed
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*x!tf_type.resource>, %[[ARG_1:[a-z0-9]*]]: tensor<*x!tf_type.resource>, %[[ARG_2:[a-z0-9]*]]: tensor<*x!tf_type.resource>)
func @replicate_resource_var_arg_shape_with_packed(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource>) {
  %0:6 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*x!tf_type.resource>, %arg2 as %rj : tensor<*x!tf_type.resource>) {n = 2: i32} {
    %1 = "tf.ReadVariableOp"(%rj) {dtype = "tfdtype$DT_FLOAT"} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    %2 = "tf.Shape"(%1) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %3 = "tf.opA"(%1, %2, %rj) : (tensor<*xf32>, tensor<?xi32>, tensor<*x!tf_type.resource>) -> tensor<*xi32>
    tf_device.return %1, %2, %3 : tensor<*xf32>, tensor<?xi32>, tensor<*xi32>
  }
  return
}

// CHECK: %[[VAR_SHAPE:[0-9]*]] = "tf.VariableShape"(%[[ARG_2]])
// CHECK: tf_device.replicate([%[[ARG_0]], %[[ARG_1]]] as %[[RI:[a-z0-9]*]]: tensor<*x!tf_type.resource>, %[[ARG_2]] as %[[RJ:[a-z0-9]*]]: tensor<*x!tf_type.resource>)
// CHECK:   %[[READ_VAR:[0-9]*]] = "tf.ReadVariableOp"(%[[RJ]])
// CHECK:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[READ_VAR]], %[[VAR_SHAPE]], %[[RJ]])
// CHECK:   tf_device.return %[[READ_VAR]], %[[VAR_SHAPE]], %[[OP_A]]

// CHECK-LABEL: func @invariant_resource_var_shape
// CHECK-SAME: (%{{[a-z0-9]*}}: tensor<*x!tf_type.resource>, %[[ARG_1:[a-z0-9]*]]: tensor<*x!tf_type.resource>)
func @invariant_resource_var_shape(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>) {
  %0 = "tf.ReadVariableOp"(%arg1) {dtype = "tfdtype$DT_FLOAT"} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
  %1:2 = tf_device.replicate([%arg0, %arg0] as %ri: tensor<*x!tf_type.resource>) {n = 2: i32} {
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


// CHECK-LABEL: func @nested_ops
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xf32>, %{{[a-z0-9]*}}: tensor<*xf32>)
func @nested_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0:8 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*xf32>) {n = 2: i32} {
    %1 = "tf.Shape"(%ri) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %2 = "tf_device.launch"() ( {
      %a = "tf.opA"(%1) : (tensor<?xi32>) -> tensor<*xi32>
      tf_device.return %a : tensor<*xi32>
    }) {device = "a"} : () -> tensor<*xi32>
    %3 = "tf_device.launch"() ( {
      %b = "tf.opB"(%1, %2) : (tensor<?xi32>, tensor<*xi32>) -> tensor<*xf32>
      tf_device.return %b : tensor<*xf32>
    }) {device = "b"} : () -> tensor<*xf32>
    %4 = "tf_device.launch"() ( {
      %c = "tf.opC"(%ri, %3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<?xi1>
      tf_device.return %c : tensor<?xi1>
    }) {device = "c"} : () -> tensor<?xi1>
    tf_device.return %1, %2, %3, %4 : tensor<?xi32>, tensor<*xi32>, tensor<*xf32>, tensor<?xi1>
  }
  return
}

// CHECK:      %[[SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_0]])
// CHECK-NEXT: %[[LAUNCH_A:[0-9]*]] = "tf_device.launch"
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[SHAPE]])
// CHECK-NEXT:   tf_device.return %[[OP_A]]
// CHECK-NEXT: device = "a"
// CHECK-NEXT: %[[LAUNCH_B:[0-9]*]] = "tf_device.launch"
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[SHAPE]], %[[LAUNCH_A]])
// CHECK-NEXT:   tf_device.return %[[OP_B]]
// CHECK-NEXT: device = "b"
// CHECK-NEXT: tf_device.replicate([{{.*}}] as %[[RI:[a-z0-9]+]]: tensor<*xf32>)
// CHECK-NEXT:   %[[LAUNCH_C:[0-9]*]] = "tf_device.launch"
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[RI]], %[[LAUNCH_B]])
// CHECK-NEXT:     tf_device.return %[[OP_C]]
// CHECK-NEXT:   device = "c"
// CHECK-NEXT:   tf_device.return %[[SHAPE]], %[[LAUNCH_A]], %[[LAUNCH_B]], %[[LAUNCH_C]]


// CHECK-LABEL:   func @do_not_hoist_ops_with_virtual_device
// CHECK-SAME:    [[VAL_0:%.*]]: tensor<*xf32>, [[VAL_1:%.*]]: tensor<*xf32>)
func @do_not_hoist_ops_with_virtual_device(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0:8 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<*xf32>) {devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2: i32} {
    %1 = "tf.Shape"(%ri) {device = "", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
    %2 = "tf.opA"(%1) {device = "TPU_REPLICATED_CORE_0"} : (tensor<?xi32>) -> tensor<*xi32>
    %3 = "tf_device.launch"() ( {
      %b = "tf.opB"(%1) : (tensor<?xi32>) -> tensor<*xi32>
      tf_device.return %b : tensor<*xi32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<*xi32>
    %4 = "tf_device.launch"() ( {
      %c = "tf.opC"(%1) {device = "TPU_REPLICATED_CORE_0"} : (tensor<?xi32>) -> tensor<*xi32>
      tf_device.return %c : tensor<*xi32>
    }) {device = "c"} : () -> tensor<*xi32>
    tf_device.return %1, %2, %3, %4 : tensor<?xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>
  }
  return
}

// CHECK:  [[SHAPE:%.*]] = "tf.Shape"([[VAL_0]])
// CHECK:  tf_device.replicate({{\[}}[[VAL_0]], [[VAL_1]]] as [[VAL_4:%.*]]: tensor<*xf32>) {devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
// CHECK:    [[OP_A:%.*]] = "tf.opA"([[SHAPE]]) {device = "TPU_REPLICATED_CORE_0"} : (tensor<?xi32>) -> tensor<*xi32>
// CHECK:    [[LAUNCH_B:%.*]] = "tf_device.launch"() ( {
// CHECK:      [[OP_B:%.*]] = "tf.opB"([[SHAPE]]) : (tensor<?xi32>) -> tensor<*xi32>
// CHECK:      tf_device.return [[OP_B]] : tensor<*xi32>
// CHECK:    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<*xi32>
// CHECK:    [[LAUNCH_C:%.*]] = "tf_device.launch"() ( {
// CHECK:      [[OP_C:%.*]] = "tf.opC"([[SHAPE]]) {device = "TPU_REPLICATED_CORE_0"} : (tensor<?xi32>) -> tensor<*xi32>
// CHECK:      tf_device.return [[OP_C]] : tensor<*xi32>
// CHECK:    }) {device = "c"} : () -> tensor<*xi32>
// CHECK:    tf_device.return [[SHAPE]], [[OP_A]], [[LAUNCH_B]], [[LAUNCH_C]]
