// RUN: tf-opt -tf-executor-update-control-dependencies %s --split-input-file --verify-diagnostics | FileCheck %s

// Test that functions must have a single graph op in their body, otherwise, the
// pass will signal failure.

// expected-error@+1 {{functions must be of a single Graph with single op Islands: function does not only contain a single tf_executor.graph}}
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

// Test that functions' graph op must have only island ops wrapping a single op
// and a single fetch op, otherwise, the pass will signal failure.

func.func @graph_multi_op_island() {
  tf_executor.graph {
    // expected-error@+1 {{functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op}}
    tf_executor.island {
      "tf.NoOp"() : () -> ()
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Test that external functions aren't supported as they're not deemed
// "export suitable".

// expected-error@+1 {{functions must be of a single Graph with single op Islands: only single block functions are supported}}
func.func private @unused_external_func()

// -----

func.func @multiple_return_no_controls_needed(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %add1, %add1_control = tf_executor.island wraps "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %add2, %add2_control = tf_executor.island wraps "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    tf_executor.fetch %add1, %add2 : tensor<*xi32>, tensor<*xi32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}
// CHECK-LABEL: func @multiple_return_no_controls_needed
// CHECK:   %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:     tf_executor.fetch %[[ADD1]], %[[ADD2]] :
// CHECK:   }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
// CHECK: }

func.func @incorrect_control_deps_replaced_with_correct_controls(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %add1, %add1_control = tf_executor.island wraps "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %add2, %add2_control = tf_executor.island(%add1_control) wraps "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %print, %print_control = tf_executor.island wraps "tf.Print"(%add2) {message = "add2 result"} : (tensor<*xi32>) -> tensor<*xi32>
    tf_executor.fetch %add1, %add2, %add2_control : tensor<*xi32>, tensor<*xi32>, !tf_executor.control
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}
// CHECK-LABEL: func @incorrect_control_deps_replaced_with_correct_controls
// CHECK:   %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:     %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD2]]) {message = "add2 result"}
// CHECK:     tf_executor.fetch %[[ADD1]], %[[ADD2]], %[[PRINT_control]] :
// CHECK:   }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
// CHECK: }

func.func @trailing_print(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %add1, %add1_control = tf_executor.island wraps "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %add2, %add2_control = tf_executor.island wraps "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %print, %print_control = tf_executor.island wraps "tf.Print"(%add2) { message = "add2 result" } : (tensor<*xi32>) -> (tensor<*xi32>)
    tf_executor.fetch %add1, %add2 : tensor<*xi32>, tensor<*xi32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}
// CHECK-LABEL: func @trailing_print
// CHECK:   %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:     %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%[[ADD2]]) {message = "add2 result"}
// CHECK:     tf_executor.fetch %[[ADD1]], %[[ADD2]], %[[PRINT_control]] :
// CHECK:   }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
// CHECK: }

func.func @non_aliasing_reads_writes(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
    %read0, %read0_control = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %assign0_control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    %read1, %read1_control = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %vh0, %vh0_control = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
    %read2, %read2_control = tf_executor.island wraps "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %assign1_control = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    %assign2_control = tf_executor.island wraps "tf.AssignVariableOp"(%arg0, %read2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    %read3, %read3_control = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    tf_executor.fetch %read3#0 : tensor<32xf32>
  }
  func.return %graph : tensor<32xf32>
}
// CHECK-LABEL: func @non_aliasing_reads_writes
// CHECK: %[[GRAPH:.*]] = tf_executor.graph {
// CHECK-DAG:   %[[READ0:.*]], %[[READ0_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg0)
// CHECK-DAG:   %[[ASSIGN0_CONTROL:.*]] = tf_executor.island(%[[READ0_CONTROL]]) wraps "tf.AssignVariableOp"(%arg0, %arg2)
// CHECK-DAG:   %[[READ1:.*]], %[[READ1_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg1)
// CHECK:   %[[VH0:.*]], %[[VH0_CONTROL:.*]] = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"}
// CHECK:   %[[READ2:.*]], %[[READ2_CONTROL:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[VH0]])
// CHECK:   %[[ASSIGN1_CONTROL:.*]] = tf_executor.island(%[[READ1_CONTROL]]) wraps "tf.AssignVariableOp"(%arg1, %[[READ0]])
// CHECK:   %[[ASSIGN2_CONTROL:.*]] = tf_executor.island(%[[ASSIGN0_CONTROL]]) wraps "tf.AssignVariableOp"(%arg0, %[[READ2]])
// CHECK:   %[[READ3:.*]], %[[READ3_CONTROL:.*]]  = tf_executor.island(%[[ASSIGN2_CONTROL]]) wraps "tf.ReadVariableOp"(%arg0)
// CHECK:   tf_executor.fetch %[[READ3]]
// CHECK: }

func.func @unknown_side_effecting_op(%arg0: tensor<32xf32>) {
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
    %outputs_2, %control_3 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %control_4 = tf_executor.island wraps "tf.AssignVariableOp"(%outputs_0, %arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    %control_5 = tf_executor.island wraps "tf._UnknownSideEffectingOp_"() : () -> ()
    %outputs_6, %control_7 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs_0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %control_8 = tf_executor.island wraps "tf.AssignVariableOp"(%outputs, %outputs_6) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    %control_9 = tf_executor.island wraps "tf.AssignVariableOp"(%outputs_0, %outputs_2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    tf_executor.fetch
  }
  return
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
// CHECK:   tf_executor.fetch

func.func @single_op_island_forward_block_arg(%arg0: tensor<?x?x?x?xbf16>) -> (tensor<2048xf32>, tensor<?x?x?x?xbf16>) {
  %0:2 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<2048xf32>} : () -> tensor<2048xf32>
    tf_executor.fetch %outputs, %arg0 : tensor<2048xf32>, tensor<?x?x?x?xbf16>
  }
  return %0#0, %0#1 : tensor<2048xf32>, tensor<?x?x?x?xbf16>
}
// CHECK-LABEL: func @single_op_island_forward_block_arg
// CHECK: tf_executor.graph {
// CHECK:   %[[outputs:.*]], %[[control:.*]] = tf_executor.island wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<2048xf32>} : () -> tensor<2048xf32>
// CHECK:   tf_executor.fetch %[[outputs]], %arg0 : tensor<2048xf32>, tensor<?x?x?x?xbf16>

func.func @tpu_load_embedding_ops_sink_controls(%arg0: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg2: tensor<*x!tf_type.resource<tensor<8xf32>>>, %arg3: tensor<*x!tf_type.resource<tensor<8xf32>>>) {
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %outputs_2, %control_3 = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %control_4 = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%outputs, %outputs_0) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table1"} : (tensor<8xf32>, tensor<8xf32>) -> ()
    %outputs_5, %control_6 = tf_executor.island wraps "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %control_7 = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%outputs_2, %outputs_5) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table2"} : (tensor<8xf32>, tensor<8xf32>) -> ()
    %control_8 = tf_executor.island wraps "tf.UnknownOp"() : () -> ()
    %control_9 = tf_executor.island wraps "tf.UnknownOp"() : () -> ()
    %control_10 = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%outputs, %outputs_0) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table3"} : (tensor<8xf32>, tensor<8xf32>) -> ()
    %control_11 = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%outputs_2, %outputs_5) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table4"} : (tensor<8xf32>, tensor<8xf32>) -> ()
    tf_executor.fetch
  }
  return
}
// CHECK-LABEL: func @tpu_load_embedding_ops_sink_controls
// CHECK:  tf_executor.graph {
// CHECK:    %[[outputs:.*]], %[[control:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK:    %[[outputs_0:.*]], %[[control_1:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK:    %[[outputs_2:.*]], %[[control_3:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK:    %[[control_4:.*]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%[[outputs]], %[[outputs_0]]) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table1"} : (tensor<8xf32>, tensor<8xf32>) -> ()
// CHECK:    %[[outputs_5:.*]], %[[control_6:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK:    %[[control_7:.*]] = tf_executor.island wraps "tf.LoadTPUEmbeddingAdagradParameters"(%[[outputs_2]], %[[outputs_5]]) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table2"} : (tensor<8xf32>, tensor<8xf32>) -> ()
// CHECK:    %[[control_8:.*]] = tf_executor.island(%[[control]], %[[control_1]], %[[control_3]], %[[control_4]], %[[control_6]], %[[control_7]]) wraps "tf.UnknownOp"() : () -> ()
// CHECK:    %[[control_9:.*]] = tf_executor.island(%[[control_8]]) wraps "tf.UnknownOp"() : () -> ()
// CHECK:    %[[control_10:.*]] = tf_executor.island(%[[control_9]]) wraps "tf.LoadTPUEmbeddingAdagradParameters"(%[[outputs]], %[[outputs_0]]) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table3"} : (tensor<8xf32>, tensor<8xf32>) -> ()
// CHECK:    %[[control_11:.*]] = tf_executor.island(%[[control_9]]) wraps "tf.LoadTPUEmbeddingAdagradParameters"(%[[outputs_2]], %[[outputs_5]]) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table4"} : (tensor<8xf32>, tensor<8xf32>) -> ()
// CHECK:    tf_executor.fetch %[[control_10]], %[[control_11]] : !tf_executor.control, !tf_executor.control

// -----

// Test that we don't create dependencies between ops on different devices, even
// if both have unknown side effects.
// Also test that the fetch op still depends on all side-effecting ops.
func.func @different_devices() {
  tf_executor.graph {
    // CHECK: %[[control:.*]] = tf_executor.island wraps "tf.A"()
    tf_executor.island wraps "tf.A"() {is_stateless = false, device = "CPU:0"} : () -> ()
    // CHECK: %[[control_2:.*]] = tf_executor.island wraps "tf.B"()
    tf_executor.island wraps "tf.B"() {is_stateless = false, device = "CPU:1"} : () -> ()
    // CHECK: tf_executor.fetch %[[control]], %[[control_2]] : !tf_executor.control, !tf_executor.control
    tf_executor.fetch
  }
  func.return
}

// -----

// Test that we do create dependencies between ops with different but compatible
// device attributes, if both ops have unknown side effects.
func.func @compatible_devices() {
  tf_executor.graph {
    // CHECK: %[[control:.*]] = tf_executor.island wraps "tf.A"()
    tf_executor.island wraps "tf.A"() {is_stateless = false, device = "CPU:0"} : () -> ()
    // CHECK: %[[control_2:.*]] =  tf_executor.island(%[[control]]) wraps "tf.B"()
    tf_executor.island wraps "tf.B"() {is_stateless = false, device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
    // CHECK: tf_executor.fetch %[[control_2]] : !tf_executor.control
    tf_executor.fetch
  }
  func.return
}

// -----

// More complex test with mixed compatible and different devices. In this case,
// side effect analysis should report following dependencies
// A -> B -> C -> D -> E -> fetch
// and we expect following dependency chains (one chain per device)
// A -> D -> fetch, B -> E -> fetch, C -> fetch.
func.func @mixed_compatible_and_different_devices() {
  tf_executor.graph {
    // CHECK: %[[control:.*]] = tf_executor.island wraps "tf.A"()
    tf_executor.island wraps "tf.A"() {is_stateless = false, device = "CPU:0"} : () -> ()
    // CHECK: %[[control_2:.*]] =  tf_executor.island wraps "tf.B"()
    tf_executor.island wraps "tf.B"() {is_stateless = false, device = "TPU:0"} : () -> ()
    // CHECK: %[[control_3:.*]] =  tf_executor.island wraps "tf.C"()
    tf_executor.island wraps "tf.C"() {is_stateless = false, device = "CPU:2"} : () -> ()
    // CHECK: %[[control_4:.*]] =  tf_executor.island(%[[control]]) wraps "tf.D"()
    tf_executor.island wraps "tf.D"() {is_stateless = false, device = "CPU:0"} : () -> ()
    // CHECK: %[[control_5:.*]] =  tf_executor.island(%[[control_2]]) wraps "tf.E"()
    tf_executor.island wraps "tf.E"() {is_stateless = false, device = "TPU:0"} : () -> ()
    // CHECK: tf_executor.fetch %[[control_3]], %[[control_4]], %[[control_5]] : !tf_executor.control, !tf_executor.control, !tf_executor.control
    tf_executor.fetch
  }
  func.return
}