// RUN: tf-opt -tf-executor-break-up-islands %s | FileCheck %s
// RUN: tf-opt -tf-executor-break-up-islands -tf-executor-break-up-islands %s | FileCheck %s

// All tests also test for idempotence.

// Test that external functions aren't processed (used to crash).
// CHECK-LABEL: func private @unused_external_func
func.func private @unused_external_func()

func.func @multiple_return(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island#0, %island#1 : tensor<*xi32>, tensor<*xi32>
  }
  func.return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @multiple_return
// CHECK:   %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:     tf_executor.fetch %[[ADD1]], %[[ADD2]] :
// CHECK:   }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
// CHECK: }

func.func @multiple_islands(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    %island2:3 = tf_executor.island(%island1#2) {
      %sub = "tf.Sub"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %mul = "tf.Mul"(%sub, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %sub, %mul : tensor<*xi32>, tensor<*xi32>
    }
    %island3 = tf_executor.island {
      %sub = "tf.Sub"(%island1#0, %island2#0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %res = "tf.Print"(%sub) { message = "sub result" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield
    }
    %island4 = tf_executor.island(%island1#2, %island2#2) {
      %add = "tf.Add"(%island1#1, %island1#1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %res = "tf.Print"(%add) { message = "add result" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield
    }
    tf_executor.fetch %island1#1, %island2#1, %island3, %island4 : tensor<*xi32>, tensor<*xi32>, !tf_executor.control, !tf_executor.control
  }
  func.return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @multiple_islands
// CHECK:  %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:    %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:    %[[SUB1:.*]], %[[SUB1_control:.*]] = tf_executor.island(%[[ADD2_control]]) wraps "tf.Sub"(%arg0, %arg1)
// CHECK:    %[[MUL:.*]], %[[MUL_control:.*]] = tf_executor.island wraps "tf.Mul"(%[[SUB1]], %arg1)
// CHECK:    %[[SUB2:.*]], %[[SUB2_control:.*]] = tf_executor.island(%[[ADD2_control]], %[[MUL_control]]) wraps "tf.Sub"(%[[ADD1]], %[[SUB1]])
// CHECK:    %[[PRINT1:.*]], %[[PRINT1_control:.*]] = tf_executor.island wraps "tf.Print"(%[[SUB2]]) {message = "sub result"}
// CHECK:    %[[ISLAND1:.*]] = tf_executor.island(%[[ADD2_control]], %[[MUL_control]]) wraps "tf.NoOp"()
// CHECK:    %[[ADD3:.*]], %[[ADD3_control:.*]] = tf_executor.island(%[[ISLAND1]], %[[ADD2_control]]) wraps "tf.Add"(%[[ADD2]], %[[ADD2]])
// CHECK:    %[[PRINT2:.*]], %[[PRINT2_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD3]]) {message = "add result"}
// CHECK:    tf_executor.fetch %[[ADD2]], %[[MUL]], %[[PRINT1_control]], %[[PRINT2_control:.*]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

func.func @dangling_print(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %res = "tf.Print"(%add2) { message = "add result" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island1#0, %island1#1 : tensor<*xi32>, tensor<*xi32>
  }
  func.return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL:  func @dangling_print
// CHECK:  %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:    %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1_control:.*]], %arg1)
// CHECK:    %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD2_control:.*]]) {message = "add result"}
// CHECK:    tf_executor.fetch %[[ADD1]], %[[ADD2]], %[[PRINT_control]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

func.func @switch_and_merge(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
  %graph:2 = tf_executor.graph {
    %island0:3 = tf_executor.island {
      %add = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %less = "tf.Less"(%arg1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %res = "tf.Print"(%add) { message = "add result 1" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add, %less : tensor<*xi32>, tensor<i1>
    }
    %switch:3 = tf_executor.Switch %island0#0, %island0#1 : tensor<*xi32>
    %island1:2 = tf_executor.island {
      %add = "tf.Add"(%switch#0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %res = "tf.Print"(%add) { message = "add result 2" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add : tensor<*xi32>
    }
    %merge_out:3 = tf_executor.Merge %island1#0, %switch#1 : tensor<*xi32>
    tf_executor.fetch %merge_out#0, %merge_out#1 : tensor<*xi32>, tensor<i32>
  }
  func.return %graph#0, %graph#1 : tensor<*xi32>, tensor<i32>
}

// CHECK-LABEL:  func @switch_and_merge(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
// CHECK: %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[LESS:.*]], %[[LESS_control:.*]] = tf_executor.island wraps "tf.Less"(%arg1, %arg1)
// CHECK:   %[[PRINT1:.*]], %[[PRINT1_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD1]]) {message = "add result 1"}
// CHECK:   %[[ISLAND1:.*]] = tf_executor.island(%[[LESS_control]], %[[PRINT1_control]]) wraps "tf.NoOp"()
// CHECK:   %[[SWITCH_false:.*]], %[[SWITCH_true:.*]], {{.*}} = tf_executor.Switch %[[ADD1]], %[[LESS]], %[[ISLAND1]]
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[SWITCH_false]], %arg1)
// CHECK:   %[[PRINT2:.*]], %[[PRINT2_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD2]]) {message = "add result 2"}
// CHECK:   %[[MERGE:.*]], %[[MERGE_index:.*]], %{{.*}} = tf_executor.Merge %[[ADD2]], %[[SWITCH_true]], %[[PRINT2_control]]
// CHECK:   tf_executor.fetch %[[MERGE]], %[[MERGE_index]]
// CHECK: }
// CHECK: return %[[GRAPH]]#0, %[[GRAPH]]#1

func.func @control_flow_plumbing(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %graph = tf_executor.graph {
    %island0:2 = tf_executor.island wraps "tf.Print"(%arg0) { message = "Random Print" } : (tensor<*xi32>) -> (tensor<*xi32>)
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add2, %island0#0 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island1#0 : tensor<*xi32>
  }
  func.return %graph : tensor<*xi32>
}

// CHECK-LABEL: func @control_flow_plumbing
// CHECK: %[[GRAPH:.*]] = tf_executor.graph {
// CHECK:   %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%arg0) {message = "Random Print"}
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island(%[[PRINT_control]]) wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:   tf_executor.fetch %[[ADD2]] : tensor<*xi32>
// CHECK: }
// CHECK: return %[[GRAPH]] : tensor<*xi32>

func.func @fetching_arg(%arg0: tensor<*xi32>) {
  tf_executor.graph {
    %island:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      tf_executor.yield %arg0, %arg0 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island#2 : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @fetching_arg
// CHECK: tf_executor.graph {
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg0)
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg0)
// CHECK:   tf_executor.fetch %[[ADD2_control]] : !tf_executor.control
// CHECK: }

func.func @non_aliasing_reads_writes(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
    %island:2 = tf_executor.island {
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %read2 = "tf.ReadVariableOp"(%var_handle) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg0, %read2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      %read3 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      tf_executor.yield %read3 : tensor<32xf32>
    }
    tf_executor.fetch %island#0 : tensor<32xf32>
  }
  func.return %graph : tensor<32xf32>
}

// CHECK-LABEL: func @non_aliasing_reads_writes
// CHECK: %[[GRAPH:.*]] = tf_executor.graph {
// CHECK:   %[[READ0:.*]], %[[READ0_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg0)
// CHECK:   %[[ASSIGN0_CONTROL:.*]] = tf_executor.island(%[[READ0_CONTROL]]) wraps "tf.AssignVariableOp"(%arg0, %arg2)
// CHECK:   %[[READ1:.*]], %[[READ1_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg1)
// CHECK:   %[[VH0:.*]], %[[VH0_CONTROL:.*]] = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"}
// CHECK:   %[[READ2:.*]], %[[READ2_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[VH0]])
// CHECK:   %[[ASSIGN1_CONTROL:.*]] = tf_executor.island(%[[READ1_CONTROL]]) wraps "tf.AssignVariableOp"(%arg1, %[[READ0:.*]])
// CHECK:   %[[ASSIGN2_CONTROL:.*]] = tf_executor.island(%[[ASSIGN0_CONTROL]]) wraps "tf.AssignVariableOp"(%arg0, %[[READ2]])
// CHECK:   %[[READ3:.*]], %[[READ3_CONTROL:.*]]  = tf_executor.island(%[[ASSIGN2_CONTROL]]) wraps "tf.ReadVariableOp"(%arg0)
// CHECK:   %[[ISLAND1:.*]] = tf_executor.island(%[[ASSIGN1_CONTROL]], %[[READ3_CONTROL]]) wraps "tf.NoOp"()
// CHECK:   tf_executor.fetch %[[READ3]], %[[ISLAND1]] : tensor<32xf32>, !tf_executor.control
// CHECK: }

func.func @unknown_side_effecting_op(%arg0: tensor<32xf32>) -> () {
  tf_executor.graph {
    %island = tf_executor.island {
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%vh1, %arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf._UnknownSideEffectingOp_"() : () -> ()
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%vh0, %read1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @unknown_side_effecting_op
// CHECK: tf_executor.graph {
// CHECK:   %[[VH0:.*]], %[[VH0_CONTROL:.*]] = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"}
// CHECK:   %[[VH1:.*]], %[[VH1_CONTROL:.*]] = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v1"}
// CHECK:   %[[READ0:.*]], %[[READ0_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[VH0]])
// CHECK:   %[[ASSIGN0_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%[[VH1]], %arg0)
// CHECK:   %[[UNKNOWN_CONTROL:.*]] = tf_executor.island(%[[READ0_CONTROL]], %[[ASSIGN0_CONTROL]]) wraps "tf._UnknownSideEffectingOp_"()
// CHECK:   %[[READ1:.*]], %[[READ1_CONTROL:.*]] = tf_executor.island(%[[UNKNOWN_CONTROL]]) wraps "tf.ReadVariableOp"(%[[VH1]])
// CHECK:   %[[ASSIGN1_CONTROL:.*]] = tf_executor.island(%[[UNKNOWN_CONTROL]]) wraps "tf.AssignVariableOp"(%[[VH0]], %[[READ1]])
// CHECK:   %[[ASSIGN2_CONTROL:.*]] = tf_executor.island(%[[READ1_CONTROL]]) wraps "tf.AssignVariableOp"(%[[VH1]], %[[READ0]])
// CHECK:   %[[ISLAND1:.*]] = tf_executor.island(%[[ASSIGN1_CONTROL]], %[[ASSIGN2_CONTROL]]) wraps "tf.NoOp"()
// CHECK:   tf_executor.fetch %[[ISLAND1]] : !tf_executor.control
// CHECK: }


// Checks empty tf_executor.island ops are populated with tf.NoOp/tf.Identity/
// tf.IdentityN ops depending on the number of data results the
// tf_executor.island has.

// CHECK-LABEL: empty_island_no_data_results
func.func @empty_island_no_data_results() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // CHECK: "tf.NoOp"
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: empty_island_single_data_result
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>)
func.func @empty_island_single_data_result(%arg0: tensor<*xf32>) {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      // CHECK: %[[IDENTITY:.*]] = "tf.Identity"
      // CHECK-SAME: (%[[ARG_0]])
      // CHECK: tf_executor.yield %[[IDENTITY]]
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: empty_island_multiple_data_results
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>, %[[ARG_1:.*]]: tensor<*xi32>)
func.func @empty_island_multiple_data_results(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      // CHECK: %[[IDENTITY_N:.*]]:2 = "tf.IdentityN"
      // CHECK-SAME: (%[[ARG_0]], %[[ARG_1]])
      // CHECK: tf_executor.yield %[[IDENTITY_N]]#0, %[[IDENTITY_N]]#1
      tf_executor.yield %arg0, %arg1 : tensor<*xf32>, tensor<*xi32>
    }
    tf_executor.fetch
  }
  func.return
}

// The following tests check that certain control dependencies between islands
// and certain tf_executor ops are added correctly.

// CHECK: %[[CONTROL:[^ ,]*]] = tf_executor.island wraps "tf.Print"
// CHECK: tf_executor.NextIteration.Sink[{{.*}}] {{.*}}, %[[CONTROL]]
func.func @next_iteration_sink_control_input() {
  tf_executor.graph {
    %source:3 = tf_executor.NextIteration.Source : tensor<*xi32>
    %island:2 = tf_executor.island {
      %const = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<*xi32>
      %print = "tf.Print"(%const) { message = "bla" } : (tensor<*xi32>) -> (tensor<*xi32>)

      tf_executor.yield %const : tensor<*xi32>
    }
    tf_executor.NextIteration.Sink[%source#1] %island#0 : tensor<*xi32>
    tf_executor.fetch %island#0 : tensor<*xi32>
  }
  func.return
}

// CHECK: %[[CONTROL:[^ ,]*]] = tf_executor.island wraps "tf.Print"
// CHECK: tf_executor.LoopCond {{.*}}, %[[CONTROL]]
func.func @loop_cond_control_input() {
  tf_executor.graph {
    %island:2 = tf_executor.island {
      %const = "tf.Const"() {value = dense<1> : tensor<i1>} : () -> tensor<*xi1>
      %print = "tf.Print"(%const) { message = "bla" } : (tensor<*xi1>) -> (tensor<*xi1>)
      tf_executor.yield %const : tensor<*xi1>
    }
    %loop_cond:2 = tf_executor.LoopCond %island#0 : tensor<*xi1>
    tf_executor.fetch %loop_cond#0 : tensor<*xi1>
  }
  func.return
}

// CHECK: %[[CONTROL:[^ ,]*]] = tf_executor.island wraps "tf.Print"
// CHECK: tf_executor.Enter {{.*}}, %[[CONTROL]]
func.func @enter_control_input() {
  tf_executor.graph {
    %island:2 = tf_executor.island {
      %const = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<*xi32>
      %print = "tf.Print"(%const) { message = "bla" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %const : tensor<*xi32>
    }
    %enter:2 = tf_executor.Enter %island#0 frame "some/frame" : tensor<*xi32>
    tf_executor.fetch %enter#0 : tensor<*xi32>
  }
  func.return
}

// CHECK: %[[CONTROL:[^ ,]*]] = tf_executor.island wraps "tf.Print"
// CHECK: tf_executor._SwitchN {{.*}}, {{.*}} of {{[0-9]*}} (%[[CONTROL]])
func.func @switchn_control_input(%arg1: tensor<i32>) {
  tf_executor.graph {
    %island:2 = tf_executor.island {
      %const = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<*xi32>
      %print = "tf.Print"(%const) { message = "bla" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %const : tensor<*xi32>
    }
    %switchn:4 = tf_executor._SwitchN %island#0, %arg1 of 3: tensor<*xi32>
    tf_executor.fetch %switchn#0 : tensor<*xi32>
  }
  func.return
}

// CHECK-LABEL: func @single_op_island_forward_block_arg
// CHECK: %[[CONST:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"
// CHECK: tf_executor.fetch %[[CONST]], %arg0
func.func @single_op_island_forward_block_arg(%arg0: tensor<?x?x?x?xbf16>) -> (tensor<2048xf32>, tensor<?x?x?x?xbf16>) {
  %0:2 = tf_executor.graph {
    %outputs:2, %control = tf_executor.island {
      %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2048xf32>} : () -> tensor<2048xf32>
      tf_executor.yield %1, %arg0 : tensor<2048xf32>, tensor<?x?x?x?xbf16>
    }
    tf_executor.fetch %outputs#0, %outputs#1 : tensor<2048xf32>, tensor<?x?x?x?xbf16>
  }
  func.return %0#0, %0#1 : tensor<2048xf32>, tensor<?x?x?x?xbf16>
}

// CHECK-LABEL: func @single_op_island_duplicate_result
// CHECK: %[[CONST:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"
// CHECK: tf_executor.fetch %[[CONST]], %[[CONST]]
func.func @single_op_island_duplicate_result() -> (tensor<2048xf32>, tensor<2048xf32>) {
  %0:2 = tf_executor.graph {
    %outputs:2, %control = tf_executor.island {
      %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2048xf32>} : () -> tensor<2048xf32>
      tf_executor.yield %1, %1 : tensor<2048xf32>, tensor<2048xf32>
    }
    tf_executor.fetch %outputs#0, %outputs#1 : tensor<2048xf32>, tensor<2048xf32>
  }
  func.return %0#0, %0#1 : tensor<2048xf32>, tensor<2048xf32>
}

// CHECK: func @tpu_load_embedding_ops_sink_controls
// CHECK: {{%.+}}, [[READ0:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: {{%.+}}, [[READ1:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: {{%.+}}, [[READ2:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: [[LOAD0:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: {{%.+}}, [[READ3:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: [[LOAD1:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: [[UNKNOWN0:%.+]] = tf_executor.island([[READ0]], [[READ1]], [[READ2]], [[LOAD0]], [[READ3]], [[LOAD1]]) wraps "tf.UnknownOp"
// CHECK: [[UNKNOWN1:%.+]] = tf_executor.island([[UNKNOWN0]]) wraps "tf.UnknownOp"
// CHECK: [[LOAD2:%.+]] = tf_executor.island([[UNKNOWN1]]) wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: [[LOAD3:%.+]] = tf_executor.island([[UNKNOWN1]]) wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: [[SINK:%.+]] = tf_executor.island([[LOAD2]], [[LOAD3]]) wraps "tf.NoOp"
// CHECK: tf_executor.fetch [[SINK]]
func.func @tpu_load_embedding_ops_sink_controls(%arg0: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg2: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg3: tensor<*x!tf_type.resource<tensor<8xf32>>>) {
 tf_executor.graph {
   %control = tf_executor.island {
     %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
     %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
     %2 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
     "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table1"} : (tensor<8xf32>, tensor<8xf32>) -> ()
     %3 = "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
     "tf.LoadTPUEmbeddingAdagradParameters"(%2, %3) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table2"} : (tensor<8xf32>, tensor<8xf32>) -> ()
     "tf.UnknownOp"() : () -> ()
     "tf.UnknownOp"() : () -> ()
     "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table3"} : (tensor<8xf32>, tensor<8xf32>) -> ()
     "tf.LoadTPUEmbeddingAdagradParameters"(%2, %3) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table4"} : (tensor<8xf32>, tensor<8xf32>) -> ()
     tf_executor.yield
   }
   tf_executor.fetch %control : !tf_executor.control
 }
 func.return
}

// CHECK: func @stateful_composite_op_control
func.func @stateful_composite_op_control(%arg0: tensor<i1>, %arg1: tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %output, %control = tf_executor.island {
      // CHECK: {{%.+}}, [[IF_CONTROL:%.+]] = tf_executor.island wraps "tf.If"
      %1 = "tf.If"(%arg0, %arg1) {device = "", else_branch = @stateful_composite_op_control_else, is_stateless = false, then_branch = @stateful_composite_op_control_then} : (tensor<i1>, tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32>
      // CHECK: [[IDENTITY_OUTPUT:%.+]], [[IDENTITY_CONTROL:%.+]] = tf_executor.island wraps "tf.Identity"
      %2 = "tf.Identity"(%1) {device = ""} : (tensor<i32>) -> tensor<i32>

      // The side effects of the If op might not be executed without an
      // explicit control dependency on the tf.If op, due to the way the
      // LowerFunctionalOpsPass in TF operates (b/185483669). Check that we
      // output an explicit control dependency on the tf.If op in this case to
      // be on the safe side.
      // CHECK: [[SINK:%.+]] = tf_executor.island([[IF_CONTROL]], [[IDENTITY_CONTROL]]) wraps "tf.NoOp"
      tf_executor.yield %2 : tensor<i32>
    }
    // CHECK: tf_executor.fetch [[IDENTITY_OUTPUT]], [[SINK]]
    tf_executor.fetch %output : tensor<i32>
  }
  func.return %0 : tensor<i32>
}

// CHECK: func @stateful_composite_op_control_else
// This is a helper function for the stateful_composite_op_control test.
func.func @stateful_composite_op_control_else(%arg0: tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island {
      %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      "tf.AssignVariableOp"(%arg0, %1) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
      tf_executor.yield %1 : tensor<i32>
    }
    tf_executor.fetch %outputs : tensor<i32>
  }
  func.return %0 : tensor<i32>
}

// CHECK: func @stateful_composite_op_control_then
// This is a helper function for the stateful_composite_op_control test.
func.func @stateful_composite_op_control_then(%arg0: tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island {
      %1 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      "tf.AssignVariableOp"(%arg0, %1) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
      tf_executor.yield %1 : tensor<i32>
    }
    tf_executor.fetch %outputs : tensor<i32>
  }
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @generator_op
func.func @generator_op(%str : tensor<!tf_type.string>, %arg0: tensor<*x!tf_type.string>, %arg1: tensor<!tf_type.string>, %arg2: tensor<*xi64>, %arg3: tensor<!tf_type.string>) {
  tf_executor.graph {
    tf_executor.island {
      // CHECK: %{{.*}}, %[[CONTROL:[^ ,]*]] = tf_executor.island wraps "tf.GeneratorDataset"
      %gen0 = "tf.GeneratorDataset"(%str, %arg0, %arg1, %arg2, %arg3) {
        finalize_func = @__finalize_func_790,
        init_func = @__init_func_530, next_func = @__next_func_680,
        next_func.experimental_ints_on_device = true,
        operand_segment_sizes = dense<[2, 2, 1]> : vector<3xi32>,
        output_shapes = [#tf_type.shape<?>],
        output_types = [f32],
        metadata = ""
      } :
         (tensor<!tf_type.string>, tensor<*x!tf_type.string>, tensor<!tf_type.string>, tensor<*xi64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.variant>
      %add1 = "tf.Add"(%str, %arg3) : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.string>
      // CHECK: tf_executor.island(%[[CONTROL]]) wraps "tf.GeneratorDataset"
      %gen1 = "tf.GeneratorDataset"(%str, %arg0, %arg1, %arg2, %arg3) {
        finalize_func = @__finalize_func_790,
        init_func = @__init_func_530, next_func = @__next_func_680,
        next_func.experimental_ints_on_device = true,
        operand_segment_sizes = dense<[2, 2, 1]> : vector<3xi32>,
        output_shapes = [#tf_type.shape<?>],
        output_types = [f32],
        metadata = ""
      } :
         (tensor<!tf_type.string>, tensor<*x!tf_type.string>, tensor<!tf_type.string>, tensor<*xi64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.variant>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}
