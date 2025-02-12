// RUN: tf-opt %s -pass-pipeline='builtin.module(tf-executor-convert-control-to-data-outputs{composite-tpuexecute-side-effects})' -split-input-file -verify-diagnostics | FileCheck %s

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Tests independent chains of two resources.

// CHECK-LABEL: func @simple_independent_chains_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>) {
func.func @simple_independent_chains_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
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
  func.return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @simple_independent_chains_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>, %[[CHAIN_1:.*]]: tensor<i32>) -> tensor<i32>
func.func @simple_independent_chains_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @simple_independent_chains
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func.func @simple_independent_chains(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[A_CONTROL:.*]] = tf_executor.island wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:6, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[A_CONTROL]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]], %[[CHAIN_CONSTANT]], %[[CHAIN_CONSTANT]])
    %while_out:4, %control_while = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @simple_independent_chains_while_body, cond = @simple_independent_chains_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: %[[B_CONTROL:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpB"() : () -> ()
    %control_B = tf_executor.island(%control_while) wraps "tf.OpB"() : () -> ()
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Tests two resources accessed by one common op (ResourceApplyAdagrad). In such
// a case we expect one common data chain for both resources.

// CHECK-LABEL: func @intersecting_chains_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>) {
func.func @intersecting_chains_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:5 = tf_executor.graph {
  %graph:4 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_2]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_1_0:.*]] = tf_executor.island(%[[CONTROL_CHAIN_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_1]], %[[ARG_3]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ADA_GRAD:.*]] = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_0]], %[[CONTROL_ASSIGN_VAR_RES_1_0]], %[[CONTROL_CHAIN_SRC]]) wraps "tf.ResourceApplyAdagrad"(%[[RES_0]], %[[RES_1]], %[[ARG_2]], %[[ARG_3]])
    %apply_grad_control = tf_executor.island(%assign_control_0, %assign_control_1) wraps "tf.ResourceApplyAdagrad"(%arg0, %arg1, %arg2, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> ()
    // CHECK: %[[LOCAL_BARRIER:.*]] = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_0]], %[[CONTROL_ASSIGN_VAR_RES_1_0]], %[[CONTROL_ADA_GRAD]]) wraps "tf.NoOp"() : () -> ()
    %local_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %apply_grad_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_0_1:.*]] = tf_executor.island(%[[LOCAL_BARRIER]], %[[CONTROL_CHAIN_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_0]], %[[ARG_3]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_2 = tf_executor.island(%local_barrier) wraps "tf.AssignVariableOp"(%arg0, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[CONTROL_ASSIGN_VAR_RES_1_1:.*]] = tf_executor.island(%[[LOCAL_BARRIER]], %[[CONTROL_CHAIN_SRC]]) wraps "tf.AssignVariableOp"(%[[RES_1]], %[[ARG_2]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_3 = tf_executor.island(%local_barrier) wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: %[[ADD:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[MUL:.*]], %{{.*}} = tf_executor.island wraps "tf.Mul"(%[[ARG_2]], %[[ARG_3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[CHAIN_SINK:.*]], %{{.*}} = tf_executor.island(%[[CONTROL_ASSIGN_VAR_RES_0_1]], %[[CONTROL_ASSIGN_VAR_RES_1_1]]) wraps "tf.Identity"(%[[CHAIN]]) : (tensor<i32>) -> tensor<i32>
    %control_barrier = tf_executor.island(%assign_control_2, %assign_control_3, %add_control, %mul_control) wraps "tf.NoOp"() : () -> ()
   // CHECK: tf_executor.fetch %[[RES_0]], %[[RES_1]], %[[ADD]], %[[MUL]], %[[CHAIN_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>
   tf_executor.fetch %arg0, %arg1, %add, %mul, %control_barrier : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2, %[[GRAPH_OUT]]#3, %[[GRAPH_OUT]]#4
  func.return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @intersecting_chains_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>, %[[CHAIN:.*]]: tensor<i32>) -> tensor<i32>
func.func @intersecting_chains_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @intersecting_chains
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func.func @intersecting_chains(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:5, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]], %[[CHAIN_CONSTANT]])
    %while_out:4, %while_control = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @intersecting_chains_while_body, cond = @intersecting_chains_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Test presence of multiple callers of a while loop body

// CHECK-LABEL: func @multiple_callers_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func.func @multiple_callers_while_body(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
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
  func.return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @multiple_callers_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func.func @multiple_callers_while_cond(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @multiple_callers
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>) {
func.func @multiple_callers(%arg0: !tf_res, %arg1: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT_0:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT_0]])
    %while_0_out:2, %while_0_control = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @multiple_callers_while_body, cond = @multiple_callers_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: %[[CONTROL_A:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island(%while_0_control) wraps "tf.OpA"() : () -> ()
    // CHECK: %[[CHAIN_CONSTANT_1:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[CONTROL_A]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT_1]])
    %while_1_out:2, %while_1_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1) {body = @multiple_callers_while_body, cond = @multiple_callers_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Test nested while ops.

// CHECK-LABEL: func @nested_loop_while_body_inner
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func.func @nested_loop_while_body_inner(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
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
  func.return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @nested_loop_while_cond_inner
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func.func @nested_loop_while_cond_inner(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: func @nested_loop_while_body_outer
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>) {
func.func @nested_loop_while_body_outer(%arg0: !tf_res, %arg1: tensor<f32>) -> (!tf_res, tensor<f32>) {
  // CHECK: %[[GRAPH_OUT:.*]]:3 = tf_executor.graph {
  %graph:2 = tf_executor.graph {
    // CHECK: %{{.*}}, %[[CONTROL_CHAIN_0_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[CONTROL_CHAIN_0_SRC]]) wraps "tf.While"(%[[RES_0]], %[[ARG_1]], %[[CHAIN_CONSTANT]])
    %while_out:2, %while_control = tf_executor.island() wraps "tf.While"(%arg0, %arg1) {body = @nested_loop_while_body_inner, cond = @nested_loop_while_cond_inner, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: %[[CHAIN_0_SINK:.*]], %{{.*}} = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.Identity"(%[[CHAIN_0]]) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_executor.fetch %[[RES_0]], %[[WHILE_OUT]]#1, %[[CHAIN_0_SINK]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<i32>
    tf_executor.fetch %arg0, %while_out#1, %while_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control
  }
  // CHECK: return %[[GRAPH_OUT]]#0, %[[GRAPH_OUT]]#1, %[[GRAPH_OUT]]#2
  func.return %graph#0, %graph#1 : !tf_res, tensor<f32>
}

// CHECK-LABEL: func @nested_loop_while_cond_outer
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_0:.*]]: tensor<i32>) -> tensor<i32>
func.func @nested_loop_while_cond_outer(%arg0: !tf_res, %arg1: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg1) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @nested_while
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<f32>) {
func.func @nested_while(%arg0: !tf_res, %arg1: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[CHAIN_CONSTANT:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[WHILE_OUT:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_CONSTANT]])
    %while_out:2, %while_control = tf_executor.island() wraps "tf.While"(%arg0, %arg1) {body = @nested_loop_while_body_outer, cond = @nested_loop_while_cond_outer, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Do not convert control outputs to chains in the presence of an op with
// unknown side effects in the while body.
// This test checks that loop signatures are unchanged and no control output is
// erased from while loop body.

// CHECK-LABEL: func @unknown_resource_op_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) {
func.func @unknown_resource_op_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
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
  func.return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @unknown_resource_op_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> tensor<i32>
func.func @unknown_resource_op_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @unknown_resource_op
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func.func @unknown_resource_op(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
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
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// No change if the no control output in while loop body.
// This test checks that loop signatures are unchanged.

// CHECK-LABEL: func @no_control_output_while_body
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) {
func.func @no_control_output_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  %graph:4 = tf_executor.graph {
    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tf_executor.fetch %arg0, %arg1, %add, %mul: tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>
  }
  func.return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @no_control_output_while_cond
// CHECK-SAME: (%[[RES_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[RES_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>, %[[ARG_3:.*]]: tensor<f32>) -> tensor<i32>
func.func @no_control_output_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @no_control_output
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<f32>>>, %[[ARG_2:.*]]: tensor<f32>) {
func.func @no_control_output(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  // CHECK: tf_executor.graph {
  tf_executor.graph {
    // CHECK: %[[A_CONTROL:.*]] = tf_executor.island wraps "tf.OpA"() : () -> ()
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    // CHECK-NOT: tf.Const
    // CHECK: %[[WHILE_OUT:.*]]:4, %[[WHILE_CONTROL:.*]] = tf_executor.island(%[[A_CONTROL]]) wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_2]])
    %while_out:4, %while_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @no_control_output_while_body, cond = @no_control_output_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    // CHECK: %[[B_CONTROL:.*]] = tf_executor.island(%[[WHILE_CONTROL]]) wraps "tf.OpB"() : () -> ()
    %control_B = tf_executor.island(%while_control) wraps "tf.OpB"() : () -> ()
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  // CHECK: return
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Tests loop with resource that is unique per iteration.
//
// In cases where a resource-allocating op creates a new unique resource per
// loop iteration (ops with `TF_UniqueResourceAllocation` trait, in this case:
// `tf.StackV2`), make sure that we don't create data dependencies between
// different iterations for such resources. This is in line with the behavior
// for the same loop unrolled. In this particular case, no data chain and token
// should be created.

func.func @unique_resource_chain(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    %while:3 = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @unique_resource_chain_while_body, cond = @unique_resource_chain_while_cond, is_stateless = false} : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}
// CHECK-LABEL:   func @unique_resource_chain
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>
// CHECK:           tf_executor.graph
// CHECK:             %[[WHILE:.*]]:2, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]]) <{body = @unique_resource_chain_while_body, cond = @unique_resource_chain_while_cond, is_stateless = false}> : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
// CHECK:             tf_executor.fetch
// CHECK:           }
// CHECK:           return

func.func @unique_resource_chain_while_body(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %graph:2 = tf_executor.graph {
    %const:2 = tf_executor.island wraps "tf.Const"() { value = dense<1000> : tensor<i32> } : () -> tensor<i32>
    %stack_handle:2 = tf_executor.island wraps "tf.StackV2"(%const#0) {elem_type = f32} : (tensor<i32>) -> !tf_res
    %stack_push:2 = tf_executor.island wraps "tf.StackPushV2"(%stack_handle#0, %arg1) : (!tf_res, tensor<f32>) -> tensor<f32>
    %add:2 = tf_executor.island wraps "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %stack_push2:2 = tf_executor.island(%stack_push#1) wraps "tf.StackPushV2"(%stack_handle#0, %add#0) : (!tf_res, tensor<f32>) -> tensor<f32>
    %one:2 = tf_executor.island wraps "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
    %add2:2 = tf_executor.island wraps "tf.Add"(%arg0, %one#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_executor.fetch %add2#0, %arg1, %stack_push2#1 : tensor<i32>, tensor<f32>, !tf_executor.control
  }
  func.return %graph#0, %graph#1 : tensor<i32>, tensor<f32>
}
// CHECK-LABEL:   func @unique_resource_chain_while_body
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>
// CHECK:           %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:             %[[THOUSAND:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1000> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[STACK_HANDLE:.*]], %{{.*}} = tf_executor.island wraps "tf.StackV2"(%[[THOUSAND]]) <{elem_type = f32}> : (tensor<i32>) -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:             %{{.*}}, %[[STACK_PUSH_CONTROL:.*]] = tf_executor.island wraps "tf.StackPushV2"(%[[STACK_HANDLE]], %[[ARG_1]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[ADD:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_1]], %[[ARG_1]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %{{.*}}, %{{.*}} = tf_executor.island(%[[STACK_PUSH_CONTROL]]) wraps "tf.StackPushV2"(%[[STACK_HANDLE]], %[[ADD]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[ONE:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[COUNTER:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_0]], %[[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:             tf_executor.fetch %[[COUNTER]], %[[ARG_1]] : tensor<i32>, tensor<f32>
// CHECK:           }
// CHECK:           return %[[GRAPH]]#0, %[[GRAPH]]#1 : tensor<i32>, tensor<f32>

func.func @unique_resource_chain_while_cond(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i1>) {
  %graph = tf_executor.graph {
    %const:2 = tf_executor.island wraps "tf.Const"() { value = dense<1000> : tensor<i32> } : () -> tensor<i32>
    %less:2 = tf_executor.island wraps "tf.Less"(%const#0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    tf_executor.fetch %less#0 : tensor<i1>
  }
  func.return %graph : tensor<i1>
}
// CHECK-LABEL:   func @unique_resource_chain_while_cond
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>
// CHECK:           %[[GRAPH:.*]] = tf_executor.graph
// CHECK:             %[[CONST:.*]], %[[CONST_CONTROL:.*]] = tf_executor.island wraps "tf.Const"() <{value = dense<1000> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[LESS:.*]], %[[LESS_CONTROL:.*]] = tf_executor.island wraps "tf.Less"(%[[CONST]], %[[ARG_0]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:             tf_executor.fetch %[[LESS]] : tensor<i1>
// CHECK:           }
// CHECK:           return %[[GRAPH]] : tensor<i1>

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Tests loop with two resource types, one of them being unique per iteration.
//
// Similar to above test but with one additional resource that is not unique per
// iteration (created by `tf.VarHandleOp`).

func.func @mixed_unique_resource_chain(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    %while:3 = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @mixed_unique_resource_chain_while_body, cond = @mixed_unique_resource_chain_while_cond, is_stateless = false} : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}
// CHECK-LABEL:   func @mixed_unique_resource_chain
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>
// CHECK:           tf_executor.graph
// CHECK:             %[[CHAIN_TOKEN:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[WHILE:.*]]:3, %[[WHILE_CONTROL:.*]] = tf_executor.island wraps "tf.While"(%[[ARG_0]], %[[ARG_1]], %[[CHAIN_TOKEN]]) <{body = @mixed_unique_resource_chain_while_body, cond = @mixed_unique_resource_chain_while_cond, is_stateless = false}> : (tensor<i32>, tensor<f32>, tensor<i32>) -> (tensor<i32>, tensor<f32>, tensor<i32>)
// CHECK:             tf_executor.fetch
// CHECK:           }
// CHECK:           return

func.func @mixed_unique_resource_chain_while_body(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %graph:2 = tf_executor.graph {
    %const:2 = tf_executor.island wraps "tf.Const"() { value = dense<1000> : tensor<i32> } : () -> tensor<i32>
    %stack_handle:2 = tf_executor.island wraps "tf.StackV2"(%const#0) {elem_type = f32} : (tensor<i32>) -> !tf_res
    %stack_push:2 = tf_executor.island wraps "tf.StackPushV2"(%stack_handle#0, %arg1) : (!tf_res, tensor<f32>) -> tensor<f32>
    %add:2 = tf_executor.island wraps "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %stack_push2:2 = tf_executor.island(%stack_push#1) wraps "tf.StackPushV2"(%stack_handle#0, %add#0) : (!tf_res, tensor<f32>) -> tensor<f32>
    %one:2 = tf_executor.island wraps "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
    %add2:2 = tf_executor.island wraps "tf.Add"(%arg0, %one#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %var_handle:2 = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
    %assign = tf_executor.island wraps "tf.AssignVariableOp"(%var_handle, %arg1) : (!tf_res, tensor<f32>) -> ()
    tf_executor.fetch %add2#0, %arg1, %stack_push2#1, %assign : tensor<i32>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1 : tensor<i32>, tensor<f32>
}
// CHECK-LABEL:   func @mixed_unique_resource_chain_while_body
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_TOKEN:.*]]: tensor<i32>
// CHECK:           %[[GRAPH:.*]]:3 = tf_executor.graph
// CHECK:             %{{.*}}, %[[CHAIN_SRC:.*]] = tf_executor.island wraps "tf.Identity"(%[[CHAIN_TOKEN]]) : (tensor<i32>) -> tensor<i32>
// CHECK:             %[[THOUSAND:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1000> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[STACK_HANDLE:.*]], %{{.*}} = tf_executor.island wraps "tf.StackV2"(%[[THOUSAND]]) <{elem_type = f32}> : (tensor<i32>) -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:             %{{.*}}, %[[STACK_PUSH_CONTROL:.*]] = tf_executor.island wraps "tf.StackPushV2"(%[[STACK_HANDLE]], %[[ARG_1]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[ADD:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_1]], %[[ARG_1]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %{{.*}}, %{{.*}} = tf_executor.island(%[[STACK_PUSH_CONTROL]]) wraps "tf.StackPushV2"(%[[STACK_HANDLE]], %[[ADD]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[ONE:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[COUNTER:.*]], %{{.*}} = tf_executor.island wraps "tf.Add"(%[[ARG_0]], %[[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:             %[[VAR_HANDLE:.*]], %{{.*}} = tf_executor.island wraps "tf.VarHandleOp"() <{container = "c", shared_name = "v0"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:             %[[ASSIGN_CONTROL:.*]] = tf_executor.island(%[[CHAIN_SRC]]) wraps "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[ARG_1]]) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:             %[[CHAIN_SINK:.*]], %{{.*}} = tf_executor.island(%[[ASSIGN_CONTROL]]) wraps "tf.Identity"(%[[CHAIN_TOKEN]]) : (tensor<i32>) -> tensor<i32>
// CHECK:             tf_executor.fetch %[[COUNTER]], %[[ARG_1]], %[[CHAIN_SINK]] : tensor<i32>, tensor<f32>, tensor<i32>
// CHECK:           }
// CHECK:           return %[[GRAPH]]#0, %[[GRAPH]]#1, %[[GRAPH]]#2 : tensor<i32>, tensor<f32>, tensor<i32>

func.func @mixed_unique_resource_chain_while_cond(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i1>) {
  %graph = tf_executor.graph {
    %const:2 = tf_executor.island wraps "tf.Const"() { value = dense<1000> : tensor<i32> } : () -> tensor<i32>
    %less:2 = tf_executor.island wraps "tf.Less"(%const#0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    tf_executor.fetch %less#0 : tensor<i1>
  }
  func.return %graph : tensor<i1>
}
// CHECK-LABEL:   func @mixed_unique_resource_chain_while_cond
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<f32>, %[[CHAIN_TOKEN:.*]]: tensor<i32>
// CHECK:           %[[GRAPH:.*]] = tf_executor.graph
// CHECK:             %[[CONST:.*]], %[[CONST_CONTROL:.*]] = tf_executor.island wraps "tf.Const"() <{value = dense<1000> : tensor<i32>}> : () -> tensor<i32>
// CHECK:             %[[LESS:.*]], %[[LESS_CONTROL:.*]] = tf_executor.island wraps "tf.Less"(%[[CONST]], %[[ARG_0]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:             tf_executor.fetch %[[LESS]] : tensor<i1>
// CHECK:           }
// CHECK:           return %[[GRAPH]] : tensor<i1>

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>

// Tests that ops not originally connected (via ctrl dep) to a fetch won't
// get a data token.

// CHECK-LABEL: func @unconnected_while_body
func.func @unconnected_while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (!tf_res, !tf_res, tensor<f32>, tensor<f32>) {
  // CHECK: tf_executor.graph
  %graph:4 = tf_executor.graph {
    // CHECK: %[[C0:.*]] = {{.*}}AssignVariableOp
    // CHECK: %[[C1:.*]] = {{.*}}AssignVariableOp
    // CHECK: VarHandleOp
    // CHECK: %[[C2:.*]] = {{.*}}AssignVariableOp
    // CHECK: %[[C3:.*]] = {{.*}}AssignVariableOp
    // CHECK: %[[SINK_0:.*]], %{{.*}} = tf_executor.island(%[[C0]]) wraps "tf.Identity"
    // CHECK: %[[EMPTY:.*]], %{{.*}} = tf_executor.island wraps "tf.Identity"
    // CHECK: %[[SINK_1:.*]], %{{.*}} = tf_executor.island(%[[C2]]) wraps "tf.Identity"
    // CHECK: tf_executor.fetch %arg0, %arg1, %arg2, %arg3, %[[SINK_0]], %[[EMPTY]], %[[SINK_1]] : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>
    %c0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (!tf_res, tensor<f32>) -> ()
    %c1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg3) : (!tf_res, tensor<f32>) -> ()
    %resource, %_ = tf_executor.island wraps "tf.VarHandleOp"() {} : () -> !tf_res
    %c2 = tf_executor.island wraps "tf.AssignVariableOp"(%resource, %arg2) : (!tf_res, tensor<f32>) -> ()
    %c3 = tf_executor.island wraps "tf.AssignVariableOp"(%resource, %arg3) : (!tf_res, tensor<f32>) -> ()
    %c2a = tf_executor.island(%c2) wraps "tf.NoOp"() : () -> ()
    %c2b = tf_executor.island(%c2a) wraps "tf.NoOp"() : () -> ()
    tf_executor.fetch %arg0, %arg1, %arg2, %arg3, %c0, %c2b : !tf_res, !tf_res, tensor<f32>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2, %graph#3 : !tf_res, !tf_res, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @unconnected_while_cond
func.func @unconnected_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL:   func @unconnected
func.func @unconnected(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  tf_executor.graph {
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    %while_out:4, %while_control = tf_executor.island(%control_A) wraps "tf.While"(%arg0, %arg1, %arg2, %arg2) {body = @unconnected_while_body, cond = @unconnected_while_cond, is_stateless = false} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, tensor<f32>)
    %control_B = tf_executor.island(%while_control) wraps "tf.OpB"() : () -> ()
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @tpu_execute_while_body
func.func @tpu_execute_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                                %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CHECK: [[exe:%.*]] = tf_executor.island({{.*}}) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg0, %key) {
        device_var_reads_indices = [0, 1],
        device_var_updates_indices = [0, 1],
        device = "task:0"
    } : (!tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK-DAG: [[exe]]{{.*}}"tf.Identity"(%arg3)
    // CHECK-DAG: "tf.Identity"(%arg4)
    // CHECK: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @tpu_execute_while_cond
func.func @tpu_execute_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @tpu_execute
func.func @tpu_execute(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @tpu_execute_while_body,
         cond = @tpu_execute_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @incomplete_composite_devices_while_body
func.func @incomplete_composite_devices_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                                %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CHECK: [[exe:%.*]] = tf_executor.island({{.*}}) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %key) {
        device_var_reads_indices = [0, 1],
        device_var_updates_indices = [0, 1],
        device = "task:0"
    } : (!tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: [[exe]]{{.*}}"tf.Identity"
    // CHECK-NOT: "tf.Identity"
    // CHECK: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @incomplete_composite_devices_while_cond
func.func @incomplete_composite_devices_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @incomplete_composite_devices
func.func @incomplete_composite_devices(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                       %arg1: !tf_res, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @incomplete_composite_devices_while_body,
         cond = @incomplete_composite_devices_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @complete_composite_devices_while_body
func.func @complete_composite_devices_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                                %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CJHECK: [[exe:%.*]] = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %key) {
        device_var_reads_indices = [0, 1],
        device_var_updates_indices = [0, 1],
        device = "task:0"
    } : (!tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"(%arg3)
    // CHECK: "tf.Identity"(%arg4)
    // CHECK: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @complete_composite_devices_while_cond
func.func @complete_composite_devices_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @complete_composite_devices
func.func @complete_composite_devices(
                       %arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                       %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @complete_composite_devices_while_body,
         cond = @complete_composite_devices_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @tpu_execute_with_non_resource_operands_while_body
func.func @tpu_execute_with_non_resource_operands_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                                %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CHECK: [[exe:%.*]] = tf_executor.island({{[^)]*}}) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg2, %arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (tensor<f32>, !tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"(%arg3)
    // CHECK: "tf.Identity"(%arg4)
    // CHECK: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @tpu_execute_with_non_resource_operands_while_cond
func.func @tpu_execute_with_non_resource_operands_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @tpu_execute_with_non_resource_operands
func.func @tpu_execute_with_non_resource_operands(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                       %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @tpu_execute_with_non_resource_operands_while_body,
         cond = @tpu_execute_with_non_resource_operands_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @double_tpu_execute_while_body
func.func @double_tpu_execute_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                         %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
    // CHECK: "tf.Identity"
  %graph:3 = tf_executor.graph {
    // CHECK: {{.*}}, [[ctrl1:%.*]] = tf_executor.island wraps "tf.Identity"
    // CHECK: {{.*}}, [[ctrl2:%.*]] = tf_executor.island wraps "tf.Identity"
    // CHECK: "tf.Identity"
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CHECK: [[exe_ctrl1:%.*]] = tf_executor.island([[ctrl1]]) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control1 = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg2, %arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (tensor<f32>, !tf_res, !tf_res, !tf_str) -> ()

    // CHECK: [[exe_ctrl2:%.*]] = tf_executor.island([[ctrl2]]) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control2 = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg2, %arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:1"
    } : (tensor<f32>, !tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control,
                                          %exe_control1, %exe_control2) wraps "tf.NoOp"() : () -> ()
    // CHECK: tf_executor.island([[exe_ctrl1]]) wraps "tf.Identity"
    // CHECK: tf_executor.island([[exe_ctrl2]]) wraps "tf.Identity"
    // CHECK: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @double_tpu_execute_while_cond
func.func @double_tpu_execute_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @double_tpu_execute
func.func @double_tpu_execute(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                              %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @double_tpu_execute_while_body,
         cond = @double_tpu_execute_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @tpu_executes_on_same_device_while_body
func.func @tpu_executes_on_same_device_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                         %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    // CHECK: "tf.Identity"
    // CHECK: {{.*}}, [[id_ctrl:%.*]] = tf_executor.island wraps "tf.Identity"
    // CHECK: "tf.Identity"
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    // CHECK: [[exe_ctrl1:%.*]] = tf_executor.island([[id_ctrl]]) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control1 = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg2, %arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (tensor<f32>, !tf_res, !tf_res, !tf_str) -> ()

    // CHECK: [[exe_ctrl2:%.*]] = tf_executor.island([[id_ctrl]]) wraps "tf.TPUExecuteAndUpdateVariables"
    %exe_control2 = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg2, %arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (tensor<f32>, !tf_res, !tf_res, !tf_str) -> ()

    %assign_control_0 = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %assign_control_1 = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %add, %add_control = tf_executor.island wraps "tf.Add"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %mul, %mul_control = tf_executor.island wraps "tf.Mul"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %control_barrier = tf_executor.island(%assign_control_0, %assign_control_1, %add_control,
                                          %exe_control1, %exe_control2) wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"(%arg3)
    // CHECK: tf_executor.island([[exe_ctrl1]], [[exe_ctrl2]]) wraps "tf.Identity"
    // CHECK: "tf.Identity"(%arg5)
    // CHECK-NEXT: tf_executor.fetch
    tf_executor.fetch %arg0, %arg1, %add, %control_barrier, %mul_control : tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @tpu_executes_on_same_device_while_cond
func.func @tpu_executes_on_same_device_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @tpu_executes_on_same_device
func.func @tpu_executes_on_same_device(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                              %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @tpu_executes_on_same_device_while_body,
         cond = @tpu_executes_on_same_device_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @tpu_execute_and_assign_variable_while_body
func.func @tpu_execute_and_assign_variable_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                         %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    // CHECK: tf.Identity
    // CHECK-NOT: tf.Identity
    // CHECK: TPUExecuteAndUpdate
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (!tf_res, !tf_res, !tf_str) -> ()

    // CHECK: AssignVariableOp
    %assign_control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) {
        device = "task:0"
    } : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK: tf.Identity
    // CHECK-NOT: tf.Identity
    %control_barrier = tf_executor.island(%assign_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: fetch
    tf_executor.fetch %arg0, %arg1, %arg2, %control_barrier : !tf_res, !tf_res, tensor<f32>, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @tpu_execute_and_assign_variable_while_cond
func.func @tpu_execute_and_assign_variable_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @tpu_execute_and_assign_variable
func.func @tpu_execute_and_assign_variable(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                              %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @tpu_execute_and_assign_variable_while_body,
         cond = @tpu_execute_and_assign_variable_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: @tpu_execute_and_assign_variable_on_different_devices_while_body
func.func @tpu_execute_and_assign_variable_on_different_devices_while_body(%arg0: !tf_res, %arg1: !tf_res,
                                         %arg2: tensor<f32>)
    -> (!tf_res, !tf_res, tensor<f32>) {
  %graph:3 = tf_executor.graph {
    // CHECK: {{.*}}, [[ctrl1:%.*]] = tf_executor.island wraps "tf.Identity"
    // CHECK: {{.*}}, [[ctrl2:%.*]] = tf_executor.island wraps "tf.Identity"
    // CHECK-NOT: tf.Identity
    // CHECK: [[exe_ctrl:%.*]] = tf_executor.island([[ctrl1]]) wraps "tf.TPUExecuteAndUpdateVariables"
    %key, %key_control = tf_executor.island wraps "tf.Const"() {value = dense<"">: !tf_str} : () -> !tf_str
    %exe_control = tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %key) {
        device_var_reads_indices = [1, 2],
        device_var_updates_indices = [1, 2],
        device = "task:0"
    } : (!tf_res, !tf_res, !tf_str) -> ()

    // CHECK: [[assign_ctrl:%.*]] = tf_executor.island([[ctrl2]]) wraps "tf.AssignVariableOp"
    %assign_control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) {
        device = "task:1"
    } : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    // CHECK-DAG: tf_executor.island([[exe_ctrl]]) wraps "tf.Identity"
    // CHECK-DAG: tf_executor.island([[assign_ctrl]]) wraps "tf.Identity"
    // CHECK-NOT: tf.Identity
    %control_barrier = tf_executor.island(%assign_control, %exe_control) wraps "tf.NoOp"() : () -> ()
    // CHECK: fetch
    tf_executor.fetch %arg0, %arg1, %arg2, %control_barrier : !tf_res, !tf_res, tensor<f32>, !tf_executor.control
  }
  func.return %graph#0, %graph#1, %graph#2 : !tf_res, !tf_res, tensor<f32>
}

// CHECK-LABEL: @tpu_execute_and_assign_variable_on_different_devices_while_cond
func.func @tpu_execute_and_assign_variable_on_different_devices_while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: tensor<f32>) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"(%arg2) : (tensor<f32>) -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @tpu_execute_and_assign_variable_on_different_devices
func.func @tpu_execute_and_assign_variable_on_different_devices(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"},
                              %arg1: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}, %arg2: tensor<f32>) {
  tf_executor.graph {
    // CHECK: "tf.Const"{{.*}}tensor<i32>
    %while_out:3, %control_while = tf_executor.island wraps "tf.While"(%arg0, %arg1, %arg2)
        {body = @tpu_execute_and_assign_variable_on_different_devices_while_body,
         cond = @tpu_execute_and_assign_variable_on_different_devices_while_cond, is_stateless = false}
    : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>)
    tf_executor.fetch
  }
  func.return
}
