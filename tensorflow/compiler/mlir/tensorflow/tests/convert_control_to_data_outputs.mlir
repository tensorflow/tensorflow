// RUN: tf-opt -tf-executor-convert-control-to-data-outputs %s | FileCheck %s

!tf_res = type tensor<!tf_type.resource<tensor<f32>>>

// Tests independent chains of two resources.

// CHECK-LABEL: func @simple_independent_chains_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>) {
func @simple_independent_chains_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:6 = tf_executor.graph {
  %graph:4 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_1_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_1]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_2]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_1:.*]] = tf_executor.island(%[[CONTROL_CHAIN_1_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_1]], %[[ARG_3]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[ADD:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[MUL:.*]], %{{.*}} = tf_executor.island wraps "tf.Mul"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CHAIN_1_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_1]]) wraps "tf.Identity"(%[[CHAIN_1]]) : (tensor<i32>) -> tensor<i32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: tf_executor.fetch %[[RES_0]], %[[RES_1]], %[[ADD]], %[[MUL]], %[[CHAIN_0_SINK]], %[[CHAIN_1_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
    tf_executor.fetch %arg0, %arg1, %add, %mul, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2, %[[GRAPH_OUT]]#3, %[[GRAPH_OUT]]#4, %[[GRAPH_OUT]]#5
  return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @simple_independent_chains_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> tensor<i32>
func @simple_independent_chains_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @simple_independent_chains
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func @simple_independent_chains(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[A_CONTROL:.*]] = tf_executor.island wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:6, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[A_CONTROL]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]], %[[CHAIN_CONSTANT]], %[[CHAIN_CONSTANT]])
    %while_out:4, %control_while = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @simple_independent_chains_while_body, cond = @simple_independent_chains_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: %[[B_CONTROL:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpB"() : () -> ()
    %control_B = tf_executor.island(%control_while) wraps "tf.OpB"() : () -> ()
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}

// Test when the two resource op chains have an op which writes to both the
// resources. ResourceApplyAdagrad access both the resources.

// CHECK-LABEL: func @intersecting_chains_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>) {
func @intersecting_chains_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:6 = tf_executor.graph {
  %graph:4 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_1_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_1]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_2]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_1_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_1_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_1]], %[[ARG_3]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ADA_GRAD:.*]] = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_0]], %[[CONTROL_ASSIGN_VAR_RES_1_0]], %[[CONTROL_CHAIN_0_SRC]], %[[CONTROL_CHAIN_1_SRC]]) wraps "tf.ResourceApplyAdagrad"(%[[RES_0]], %[[RES_1]], %[[ARG_2]], %[[ARG_3]])
    %apply_grad_control = tf_executor.island(%assign_control_0, %assign_control_1) wraps "tf.ResourceApplyAdagrad"(%arg0, %arg1, %arg2, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> ()
    // CHECK: %[[LOCAL_BARRIER:.*]] = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_0]], %[[CONTROL_ASSIGN_VAR_RES_1_0]], %[[CONTROL_ADA_GRAD]]) wraps "tf.NoOp"() : () -> ()
    %local_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %apply_grad_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0_1:.*]] = tf_executor.island(%[[LOCAL_BARRIER]], %[[CONTROL_CHAIN_0_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_3]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_2 = tf_executor.island(%local_barrier) wraps "tf.AssignVariableOp"(%arg0, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_1_1:.*]] = tf_executor.island(%[[LOCAL_BARRIER]], %[[CONTROL_CHAIN_1_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_1]], %[[ARG_2]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_3 = tf_executor.island(%local_barrier) wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[ADD:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[MUL:.*]], %{{.*}} = tf_executor.island wraps "tf.Mul"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_0]], %[[CONTROL_ADA_GRAD]], %[[CONTROL_ASSIGN_VAR_RES_0_1]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CHAIN_1_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_1_0]], %[[CONTROL_ADA_GRAD]], %[[CONTROL_ASSIGN_VAR_RES_1_1]]) wraps "tf.Identity"(%[[CHAIN_1]]) : (tensor<i32>) -> tensor<i32>
    %control_barrier = tf_executor.island(%assign_control_2, %assign_control_3, %add_control, %mul_control) wraps "tf.NoOp"() : () -> ()
   // CHECK: tf_executor.fetch %[[RES_0]], %[[RES_1]], %[[ADD]], %[[MUL]], %[[CHAIN_0_SINK]], %[[CHAIN_1_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
   tf_executor.fetch %arg0, %arg1, %add, %mul, %control_barrier : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2, %[[GRAPH_OUT]]#3, %[[GRAPH_OUT]]#4, %[[GRAPH_OUT]]#5
  return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @intersecting_chains_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> tensor<i32>
func @intersecting_chains_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @intersecting_chains
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func @intersecting_chains(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:6, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]], %[[CHAIN_CONSTANT]], %[[CHAIN_CONSTANT]])
    %while_out:4, %while_control = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @intersecting_chains_while_body, cond = @intersecting_chains_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}

// Test presence of multiple callers of a while loop body

// CHECK-LABEL: func @multiple_callers_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func @multiple_callers_while_body(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:3 = tf_executor.graph {
  %graph:2 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_1]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_executor.fetch %[[RES_0]], %[[ARG_1]], %[[CHAIN_0_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>
    tf_executor.fetch %arg0, %arg1, %control : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2
  return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @multiple_callers_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func @multiple_callers_while_cond(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @multiple_callers
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>) {
func @multiple_callers(%arg0: !tf_res, %arg1: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT_0:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT_0]])
    %while_0_out:2, %while_0_control = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @multiple_callers_while_body, cond = @multiple_callers_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: %[[CONTROL_A:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island(%while_0_control) wraps "tf.OpA"() : () -> ()
    // CHECK: %[[CHAIN_CONSTANT_1:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[CONTROL_A]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT_1]])
    %while_1_out:2, %while_1_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1) {body = @multiple_callers_while_body, cond = @multiple_callers_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}

// Test nested while ops.

// CHECK-LABEL: func @nested_loop_while_body_inner
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func @nested_loop_while_body_inner(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:3 = tf_executor.graph {
  %graph:2 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_1]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_executor.fetch %[[RES_0]], %[[ARG_1]], %[[CHAIN_0_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>
    tf_executor.fetch %arg0, %arg1, %control : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2
  return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @nested_loop_while_cond_inner
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func @nested_loop_while_cond_inner(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL: func @nested_loop_while_body_outer
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func @nested_loop_while_body_outer(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:3 = tf_executor.graph {
  %graph:2 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.While"(%[[RES_0]], %[[ARG_1]], %[[CHAIN_CONSTANT]])
    %while_out:2, %while_control = tf_executor.island() wraps "tf.While"(%arg0, %arg1) {body = @nested_loop_while_body_inner, cond = @nested_loop_while_cond_inner, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_executor.fetch %[[RES_0]], %[[WHILE_OUT]]#1, %[[CHAIN_0_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>
    tf_executor.fetch %arg0, %while_out#1, %while_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2
  return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @nested_loop_while_cond_outer
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func @nested_loop_while_cond_outer(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @nested_while
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>) {
func @nested_while(%arg0: !tf_res, %arg1: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT]])
    %while_out:2, %while_control = tf_executor.island() wraps "tf.While"(%arg0, %arg1) {body = @nested_loop_while_body_outer, cond = @nested_loop_while_cond_outer, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}

// Do not convert control outputs to chains in the presence of an op with
// unknown side effects in the while body.
// This test checks that loop signatures are unchanged and no control output is
// erased from while loop body.

// CHECK-LABEL: func @unknown_resource_op_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) {
func @unknown_resource_op_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  %graph:4 = tf_executor.graph {
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %control_unknown = tf_executor.island wraps "tf.UnknownOp"() : () -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control, %mul_control) wraps "tf.NoOp"() : () -> ()
    // Checks fetch op is not modified.
    // CHECK: tf_executor.fetch
    // CHECK-SAME: tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, !tf_executor.control
    tf_executor.fetch %arg0, %arg1, %add, %mul, %control_barrier: tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, !tf_executor.control
  }
  return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @unknown_resource_op_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> tensor<i32>
func @unknown_resource_op_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @unknown_resource_op
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func @unknown_resource_op(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[A_CONTROL:.*]] = tf_executor.island wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    // CHECK-NOT: tf.Const
    // CHECK: %[[WHILE_OUT:.*]]:4, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[A_CONTROL]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]])
    %while_out:4, %while_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @unknown_resource_op_while_body, cond = @unknown_resource_op_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: %[[B_CONTROL:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpB"() : () -> ()
    %control_B = tf_executor.island(%while_control) wraps "tf.OpB"() : () -> ()
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}

// No change if the no control output in while loop body.
// This test checks that loop signatures are unchanged.

// CHECK-LABEL: func @no_control_output_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) {
func @no_control_output_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  %graph:4 = tf_executor.graph {
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tf_executor.fetch %arg0, %arg1, %add, %mul: tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>
  }
  return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @no_control_output_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> tensor<i32>
func @no_control_output_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  return %graph : tensor<i32>
}

// CHECK-LABEL:   func @no_control_output
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func @no_control_output(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[A_CONTROL:.*]] = tf_executor.island wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    // CHECK-NOT: tf.Const
    // CHECK: %[[WHILE_OUT:.*]]:4, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[A_CONTROL]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]])
    %while_out:4, %while_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @unknown_resource_op_while_body, cond = @unknown_resource_op_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: %[[B_CONTROL:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpB"() : () -> ()
    %control_B = tf_executor.island(%while_control) wraps "tf.OpB"() : () -> ()
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  return
}
