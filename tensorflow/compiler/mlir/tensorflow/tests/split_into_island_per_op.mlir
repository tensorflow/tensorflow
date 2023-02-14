// RUN: tf-opt -tf-executor-split-into-island-per-op %s --split-input-file --verify-diagnostics | FileCheck %s

// All tests also test for idempotence.

// Test that functions must have a single graph op in their body, otherwise, the
// pass will signal failure.

// expected-error@+1 {{expected function to contain only a graph_op}}
func.func @multiple_func_body_ops() {
  tf_executor.graph {
    tf_executor.island {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  tf_executor.graph {
    tf_executor.island {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Test that functions' graph op must have a single island op and a single fetch
// op, otherwise, the pass will signal failure.

func.func @graph_multiple_islands() {
  // expected-error@+1 {{expected graph op to contain only a single island_op and a single fetch_op}}
  tf_executor.graph {
    tf_executor.island {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    tf_executor.island {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Test that V1 control flow is not allowed in this pass. Otherwise, if found,
// the pass should signal failure.

func.func @switch_and_merge(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
  // expected-error@below {{expected graph op to contain only a single island_op and a single fetch_op}}
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

// -----

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
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:    %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD2]]) {message = "add result"}
// CHECK:    tf_executor.fetch %[[ADD1]], %[[ADD2]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

func.func @drop_fetch_control_dep(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island1#0, %island1#1, %island1#2 : tensor<*xi32>, tensor<*xi32>, !tf_executor.control
  }
  func.return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL:  func @drop_fetch_control_dep
// CHECK:  %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:    %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:    tf_executor.fetch %[[ADD1]], %[[ADD2]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

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
// CHECK:   tf_executor.fetch
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
// CHECK:   %[[ASSIGN0_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2)
// CHECK:   %[[READ1:.*]], %[[READ1_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg1)
// CHECK:   %[[VH0:.*]], %[[VH0_CONTROL:.*]] = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"}
// CHECK:   %[[READ2:.*]], %[[READ2_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[VH0]])
// CHECK:   %[[ASSIGN1_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %[[READ0]])
// CHECK:   %[[ASSIGN2_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %[[READ2]])
// CHECK:   %[[READ3:.*]], %[[READ3_CONTROL:.*]]  = tf_executor.island wraps "tf.ReadVariableOp"(%arg0)
// CHECK:   tf_executor.fetch %[[READ3]]
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
// CHECK:   %[[UNKNOWN_CONTROL:.*]] = tf_executor.island wraps "tf._UnknownSideEffectingOp_"()
// CHECK:   %[[READ1:.*]], %[[READ1_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[VH1]])
// CHECK:   %[[ASSIGN1_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%[[VH0]], %[[READ1]])
// CHECK:   %[[ASSIGN2_CONTROL:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%[[VH1]], %[[READ0]])
// CHECK:   tf_executor.fetch

// Checks empty tf_executor.island ops are populated with tf.NoOp/tf.Identity/
// tf.IdentityN ops depending on the number of data results the
// tf_executor.island has.

func.func @empty_island_no_data_results() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}
// CHECK-LABEL: empty_island_no_data_results
// CHECK: "tf.NoOp"

func.func @empty_island_single_data_result(%arg0: tensor<*xf32>) {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch
  }
  func.return
}
// CHECK-LABEL: empty_island_single_data_result
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>)
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"
// CHECK-SAME: (%[[ARG_0]])
// CHECK: tf_executor.yield %[[IDENTITY]]

func.func @empty_island_multiple_data_results(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      tf_executor.yield %arg0, %arg1 : tensor<*xf32>, tensor<*xi32>
    }
    tf_executor.fetch
  }
  func.return
}
// CHECK-LABEL: empty_island_multiple_data_results
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>, %[[ARG_1:.*]]: tensor<*xi32>)
// CHECK: %[[IDENTITY_N:.*]]:2 = "tf.IdentityN"
// CHECK-SAME: (%[[ARG_0]], %[[ARG_1]])
// CHECK: tf_executor.yield %[[IDENTITY_N]]#0, %[[IDENTITY_N]]#1

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
// CHECK-LABEL: func @single_op_island_forward_block_arg
// CHECK: %[[CONST:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"
// CHECK: tf_executor.fetch %[[CONST]], %arg0

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
// CHECK-LABEL: func @single_op_island_duplicate_result
// CHECK: %[[CONST:.*]], %{{.*}} = tf_executor.island wraps "tf.Const"
// CHECK: tf_executor.fetch %[[CONST]], %[[CONST]]

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
// CHECK-LABEL: func @tpu_load_embedding_ops_sink_controls
// CHECK: {{%.+}}, [[READ0:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: {{%.+}}, [[READ1:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: {{%.+}}, [[READ2:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: [[LOAD0:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: {{%.+}}, [[READ3:%.+]] = tf_executor.island wraps "tf.ReadVariableOp"
// CHECK: [[LOAD1:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: [[UNKNOWN0:%.+]] = tf_executor.island wraps "tf.UnknownOp"
// CHECK: [[UNKNOWN1:%.+]] = tf_executor.island wraps "tf.UnknownOp"
// CHECK: [[LOAD2:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: [[LOAD3:%.+]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"
// CHECK: tf_executor.fetch



// CHECK-LABEL: func @stateful_composite_op_control
func.func @stateful_composite_op_control(%arg0: tensor<i1>, %arg1: tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %output, %control = tf_executor.island {
      // CHECK: {{%.+}}, [[IF_CONTROL:%.+]] = tf_executor.island wraps "tf.If"
      %1 = "tf.If"(%arg0, %arg1) {device = "", else_branch = @stateful_composite_op_control_else, is_stateless = false, then_branch = @stateful_composite_op_control_then} : (tensor<i1>, tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32>
      // CHECK: [[IDENTITY_OUTPUT:%.+]], [[IDENTITY_CONTROL:%.+]] = tf_executor.island wraps "tf.Identity"
      %2 = "tf.Identity"(%1) {device = ""} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %2 : tensor<i32>
    }
    // CHECK: tf_executor.fetch [[IDENTITY_OUTPUT]]
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
        operand_segment_sizes = array<i32: 2, 2, 1>,
        output_shapes = [#tf_type.shape<?>],
        output_types = [f32],
        metadata = ""
      } :
         (tensor<!tf_type.string>, tensor<*x!tf_type.string>, tensor<!tf_type.string>, tensor<*xi64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.variant>
      // CHECK: %{{.*}}, %[[CONTROL2:[^ ,]*]] = tf_executor.island wraps "tf.Add"
      %add1 = "tf.Add"(%str, %arg3) : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.string>
      // CHECK: %{{.*}}, %[[CONTROL3:[^ ,]*]] = tf_executor.island wraps "tf.GeneratorDataset"
      %gen1 = "tf.GeneratorDataset"(%str, %arg0, %arg1, %arg2, %arg3) {
        finalize_func = @__finalize_func_790,
        init_func = @__init_func_530, next_func = @__next_func_680,
        next_func.experimental_ints_on_device = true,
        operand_segment_sizes = array<i32: 2, 2, 1>,
        output_shapes = [#tf_type.shape<?>],
        output_types = [f32],
        metadata = ""
      } :
         (tensor<!tf_type.string>, tensor<*x!tf_type.string>, tensor<!tf_type.string>, tensor<*xi64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.variant>
      tf_executor.yield
    }
    // CHECK: tf_executor.fetch
    tf_executor.fetch
  }
  func.return
}

func.func @then_function() {
  tf_executor.graph {
    tf_executor.island {
      "tf.OpA"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

func.func @else_function() {
  tf_executor.graph {
    tf_executor.island {
      "tf.OpB"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}