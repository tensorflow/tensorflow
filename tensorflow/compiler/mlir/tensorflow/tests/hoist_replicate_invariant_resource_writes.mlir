// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-hoist-replicate-invariant-resource-writes | FileCheck %s

!tf_res_i32 = type tensor<*x!tf_type.resource<tensor<i32>>>
!tf_res_f32 = type tensor<*x!tf_type.resource<tensor<f32>>>

// CHECK-LABEL: func @hoist_tail_assign
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>)
func @hoist_tail_assign(%arg0: !tf_res_f32) {
  // CHECK: [[REPLICATE:%.*]]:4 = tf_device.replicate {n = 2 : i32}
  %replicate:2 = tf_device.replicate {n = 2 : i32} {
    // CHECK: [[OP_A:%.*]] = "tf.OpA"
    %op_a = "tf.OpA"() : () -> tensor<f32>
    // CHECK: [[OP_B:%.*]] = "tf.OpB"
    %op_b = "tf.OpB"() : () -> tensor<i32>
    // CHECK-NOT: tf.AssignVariableOp
    "tf.AssignVariableOp"(%arg0, %op_a) : (!tf_res_f32, tensor<f32>) -> ()
    // CHECK: tf_device.return [[OP_B]], [[OP_A]] : tensor<i32>, tensor<f32>
    tf_device.return %op_b : tensor<i32>
  }
  // CHECK: "tf.AssignVariableOp"([[ARG0]], [[REPLICATE]]#2)
  return
}

// CHECK-LABEL: func @do_not_hoist_non_tail_assigns
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>)
func @do_not_hoist_non_tail_assigns(%arg0: !tf_res_f32) {
  // CHECK: tf_device.replicate {n = 2 : i32}
  tf_device.replicate {n = 2 : i32} {
    // CHECK: [[OP_A:%.*]] = "tf.OpA"
    %op_a = "tf.OpA"() : () -> tensor<f32>
    // CHECK: "tf.AssignVariableOp"([[ARG0]], [[OP_A]])
    "tf.AssignVariableOp"(%arg0, %op_a) : (!tf_res_f32, tensor<f32>) -> ()
    // CHECK: "tf.ResourceAccessOp"([[ARG0]])
    "tf.ResourceAccessOp"(%arg0) : (!tf_res_f32) -> tensor<i32>
    // CHECK: tf_device.return
    tf_device.return
  }
  // CHECK-NOT: tf.AssignVariableOp
  return
}


// CHECK-LABEL: func @do_not_hoist_writes_to_explicitly_captured_resources
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>, [[ARG1:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>) {
func @do_not_hoist_writes_to_explicitly_captured_resources(%arg0: !tf_res_f32, %arg1: !tf_res_f32) {
  // CHECK: [[REPLICATE:%.*]]:2 = tf_device.replicate({{\[}}[[ARG0]], [[ARG1]]] as [[RI:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>) {n = 2 : i32}
  %replicate:2 = tf_device.replicate ([%arg0, %arg1] as %ri: tensor<*x!tf_type.resource<tensor<f32>>>) {n = 2 : i32} {
    // CHECK: [[OP_A:%.*]] = "tf.OpA"()
    %op_a = "tf.OpA"() : () -> tensor<f32>
    // CHECK: [[OP_B:%.*]] = "tf.OpB"()
    %op_b = "tf.OpB"() : () -> tensor<i32>
    // CHECK: "tf.AssignVariableOp"([[RI]], [[OP_A]])
    "tf.AssignVariableOp"(%ri, %op_a) : (!tf_res_f32, tensor<f32>) -> ()
    // CHECK: tf_device.return [[OP_B]] : tensor<i32>
    tf_device.return %op_b : tensor<i32>
  }
  // CHECK-NOT: tf.AssignVariableOp
  return
}

// CHECK-LABEL: func @only_hoist_tail_assign
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>)
func @only_hoist_tail_assign(%arg0: !tf_res_f32) {
  // CHECK: [[REPLICATE:%.*]]:2 = tf_device.replicate {n = 2 : i32}
  tf_device.replicate {n = 2 : i32} {
    // CHECK: [[OP_A:%.*]] = "tf.OpA"
    %op_a = "tf.OpA"() : () -> tensor<f32>
    // CHECK: [[OP_B:%.*]] = "tf.OpB"
    %op_b = "tf.OpB"() : () -> tensor<f32>
    // CHECK: "tf.AssignVariableOp"([[ARG0]], [[OP_A]])
    "tf.AssignVariableOp"(%arg0, %op_a) : (!tf_res_f32, tensor<f32>) -> ()
    // CHECK-NOT: tf.AssignVariableOp
    "tf.AssignVariableOp"(%arg0, %op_b) : (!tf_res_f32, tensor<f32>) -> ()
    // CHECK: tf_device.return [[OP_B]] : tensor<f32>
    tf_device.return
  }
  // CHECK: "tf.AssignVariableOp"([[ARG0]], [[REPLICATE]]#0)
  return
}
