// RUN: tf-opt -split-input-file -tf-test-side-effect-analysis -verify-diagnostics %s | FileCheck %s --dump-input=fail

// Tests that the pass tracks control dependencies for reads/writes on the same
// resource.

// CHECK-LABEL: func @non_aliasing_reads_writes
func @non_aliasing_reads_writes(
// expected-remark@above {{ID: 13}}
// expected-remark@above {{Predecessors: {12}}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Predecessors: {10}}}
  // expected-remark@above {{Successors: {12}}}
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
    // expected-remark@above {{Successors: {10}}}
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {1}}}
      "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {6}}}
      %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Successors: {5}}}
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 3}}
      %read2 = "tf.ReadVariableOp"(%var_handle) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%arg1, %read0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%arg0, %read2) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {7}}}
      %read3 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {6}}}
      // expected-remark@above {{Successors: {8}}}
      tf_executor.yield %read3 : tensor<32xf32>
      // expected-remark@above {{ID: 8}}
      // expected-remark@above {{Predecessors: {4,5,7}}}
      // expected-remark@above {{Successors: {9}}}
    }
    tf_executor.fetch %island#0 : tensor<32xf32>
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Predecessors: {9}}}
    // expected-remark@above {{Successors: {11}}}
  }
  return %graph : tensor<32xf32>
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Predecessors: {11}}}
  // expected-remark@above {{Successors: {13}}}
}

// -----

// Tests that the pass tracks control dependencies for reads/writes on the two
// resource handles that refer to the same variable.

// CHECK-LABEL: func @aliasing_reads_writes
func @aliasing_reads_writes(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 14}}
// expected-remark@above {{Predecessors: {13}}}
  tf_executor.graph {
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Predecessors: {11}}}
  // expected-remark@above {{Successors: {13}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Predecessors: {9}}}
    // expected-remark@above {{Successors: {11}}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %vh1_id:2 = "tf.IdentityN"(%vh1, %arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>)
      // expected-remark@above {{ID: 2}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1_id#0, %arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {3}}}
      // expected-remark@above {{Successors: {5,6}}}
      %read1 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
      %read2 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
      "tf.AssignVariableOp"(%vh0, %read2) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%vh1_id#0, %read1) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 8}}
      // expected-remark@above {{Predecessors: {7}}}
      // expected-remark@above {{Successors: {9}}}
      tf_executor.yield
      // expected-remark@above {{ID: 9}}
      // expected-remark@above {{Predecessors: {8}}}
      // expected-remark@above {{Successors: {10}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 11}}
    // expected-remark@above {{Predecessors: {10}}}
    // expected-remark@above {{Successors: {12}}}
  }
  return
  // expected-remark@above {{ID: 13}}
  // expected-remark@above {{Predecessors: {12}}}
  // expected-remark@above {{Successors: {14}}}
}

// -----

// Tests that the pass tracks control dependencies for side-effecting on unknown
// resources.

// CHECK-LABEL: func @unknown_side_effecting_op
func @unknown_side_effecting_op(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 13}}
// expected-remark@above {{Predecessors: {12}}}
  tf_executor.graph {
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Predecessors: {10}}}
  // expected-remark@above {{Successors: {12}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
    // expected-remark@above {{Successors: {10}}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1, %arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Successors: {4}}}
      "tf._UnknownSideEffectingOp_"() : () -> ()
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {2,3}}}
      // expected-remark@above {{Successors: {5,6}}}
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
      "tf.AssignVariableOp"(%vh0, %read1) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5}}}
      // expected-remark@above {{Successors: {8}}}
      tf_executor.yield
      // expected-remark@above {{ID: 8}}
      // expected-remark@above {{Predecessors: {6,7}}}
      // expected-remark@above {{Successors: {9}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Predecessors: {9}}}
    // expected-remark@above {{Successors: {11}}}
  }
  return
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Predecessors: {11}}}
  // expected-remark@above {{Successors: {13}}}
}

// -----

// Tests that the pass tracks control dependencies for read-only ops on unknown
// resources.

// CHECK-LABEL: func @read_only_unknown_resource
func @read_only_unknown_resource(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 10}}
// expected-remark@above {{Predecessors: {9}}}
  tf_executor.graph {
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Predecessors: {7}}}
  // expected-remark@above {{Successors: {9}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
    // expected-remark@above {{Successors: {7}}}
      %vh0 = "tf._UnknownSideEffectingOp_"() : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2,3}}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {4}}}
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {2,3}}}
      // expected-remark@above {{Successors: {5}}}
      tf_executor.yield
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 7}}
    // expected-remark@above {{Predecessors: {6}}}
    // expected-remark@above {{Successors: {8}}}
  }
  return
  // expected-remark@above {{ID: 9}}
  // expected-remark@above {{Predecessors: {8}}}
  // expected-remark@above {{Successors: {10}}}
}
