// RUN: tf-opt -split-input-file -tf-test-side-effect-analysis -verify-diagnostics %s | FileCheck %s --dump-input=fail

// Tests that the pass tracks control dependencies for reads/writes on the same
// resource.

// CHECK-LABEL: func @non_aliasing_reads_writes
func @non_aliasing_reads_writes(
// expected-remark@above {{ID: 13}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Successors: {12}}}
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 9}}
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
    }
    tf_executor.fetch %island#0 : tensor<32xf32>
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Predecessors: {9}}}
  }
  return %graph : tensor<32xf32>
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Predecessors: {11}}}
}

// -----

// Tests that the pass tracks control dependencies for reads/writes on the two
// resource handles that refer to the same variable.

// CHECK-LABEL: func @aliasing_reads_writes
func @aliasing_reads_writes(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 14}}
  tf_executor.graph {
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Successors: {13}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 10}}
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
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 11}}
    // expected-remark@above {{Predecessors: {10}}}
  }
  return
  // expected-remark@above {{ID: 13}}
  // expected-remark@above {{Predecessors: {12}}}
}

// -----

// Tests that the pass tracks control dependencies for side-effecting on unknown
// resources.

// CHECK-LABEL: func @unknown_side_effecting_op
func @unknown_side_effecting_op(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 14}}
  tf_executor.graph {
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Successors: {13}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Successors: {11}}}
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
      // expected-remark@above {{Successors: {5,6,7}}}
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {8}}}
      %read2 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%vh0, %read1) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {9}}}
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 8}}
      // expected-remark@above {{Predecessors: {5,6}}}
      // expected-remark@above {{Successors: {9}}}
      tf_executor.yield
      // expected-remark@above {{ID: 9}}
      // expected-remark@above {{Predecessors: {7,8}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 11}}
    // expected-remark@above {{Predecessors: {10}}}
  }
  return
  // expected-remark@above {{ID: 13}}
  // expected-remark@above {{Predecessors: {12}}}
}

// -----

// Tests that the pass tracks control dependencies for read-only ops on unknown
// resources.

// CHECK-LABEL: func @read_only_unknown_resource
func @read_only_unknown_resource(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 10}}
  tf_executor.graph {
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Successors: {9}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 6}}
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
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 7}}
    // expected-remark@above {{Predecessors: {6}}}
  }
  return
  // expected-remark@above {{ID: 9}}
  // expected-remark@above {{Predecessors: {8}}}
}

// -----

// Tests that the pass adds control dependencies in nested regions with
// tf_device.replicate

func @with_replicate(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg3: tensor<*x!tf.resource<tensor<32xf32>>>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
  // expected-remark@above {{Successors: {11}}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Successors: {9}}}
      %u0:2 = "tf._UnknownSideEffectingOp_"() : () -> (tensor<32xf32>, tensor<32xf32>)
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {5}}}
      tf_device.replicate(
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {6}}}
          [%arg0, %arg1] as %r0: tensor<*x!tf.resource<tensor<32xf32>>>,
          [%arg2, %arg3] as %r1: tensor<*x!tf.resource<tensor<32xf32>>>,
          [%u0#0, %u0#1] as %u : tensor<32xf32>)
          {n = 2 : i32, devices = ["/CPU:0", "/GPU:1"]} {
        %read0 = "tf.ReadVariableOp"(%r0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {4}}}
        "tf.AssignVariableOp"(%r1, %u) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        %read1 = "tf.ReadVariableOp"(%r1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
        // expected-remark@above {{Successors: {4}}}
        tf_device.return
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Predecessors: {1,3}}}
      }
      "tf._UnknownSideEffectingOp_"() : () -> ()
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {5}}}
      // expected-remark@above {{Successors: {7}}}
      tf_executor.yield
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
  }
  return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Predecessors: {10}}}
}

// -----

// Tests that the pass does not add control dependencies a stateless if op.

// CHECK-LABEL: func @stateless_if_op
func @stateless_if_op(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Successors: {7}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %if = "tf.If"(%arg1, %arg1) {
      // expected-remark@above {{ID: 1}}
          then_branch = @if_then, else_branch = @if_else, is_stateless = true}
        : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Predecessors: {6}}}
}

// CHECK-LABEL: func @if_then
func @if_then(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
  // expected-remark@above {{Successors: {4}}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
    // expected-remark@above {{Successors: {2}}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
    // expected-remark@above {{Predecessors: {1}}}
  }
  return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Predecessors: {3}}}
}

// CHECK-LABEL: func @if_else
func @if_else(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
  // expected-remark@above {{Successors: {4}}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
    // expected-remark@above {{Successors: {2}}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
    // expected-remark@above {{Predecessors: {1}}}
  }
  return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Predecessors: {3}}}
}

// -----

// Tests that the pass does not add control dependencies a stateless while op.

// CHECK-LABEL: func @stateless_if_op
func @stateless_if_op(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Successors: {7}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %if = "tf.While"(%arg1) {
      // expected-remark@above {{ID: 1}}
          body = @while_body, cond = @while_cond, is_stateless = true}
        : (tensor<i1>) -> tensor<i1>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Predecessors: {6}}}
}

// CHECK-LABEL: func @while_body
func @while_body(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
  // expected-remark@above {{Successors: {4}}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
    // expected-remark@above {{Successors: {2}}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
    // expected-remark@above {{Predecessors: {1}}}
  }
  return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Predecessors: {3}}}
}

// CHECK-LABEL: func @while_cond
func @while_cond(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
  // expected-remark@above {{Successors: {4}}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
    // expected-remark@above {{Successors: {2}}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
    // expected-remark@above {{Predecessors: {1}}}
  }
  return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Predecessors: {3}}}
}

// -----

// Tests that the pass tracks control dependencies for variables from an if op's
// output.

// CHECK-LABEL: func @output_of_if_op
func @output_of_if_op(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
  // expected-remark@above {{Successors: {11}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Successors: {9}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      %if:3 = "tf.If"(%arg2, %id0, %arg1) {
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2,3,4}}}
          then_branch = @if_then, else_branch = @if_else, is_stateless = false}
        : (tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>) ->
          (tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%if#0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5,6}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%if#1) :
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%if#2) :
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5,6}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2,3,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {2,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
  }
  return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Predecessors: {10}}}
}

// CHECK-LABEL: func @if_then
func @if_then(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>) ->
  (tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>) {
  %graph:3 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
  // expected-remark@above {{Successors: {6}}}
    %island:4 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %u0 = "tf._UnknownSideEffectingOp_"() : ()
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 1}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      tf_executor.yield %u0, %id0, %id0 :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
        tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2 :
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
      tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>
  }
  return %graph#0, %graph#1, %graph#2 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Predecessors: {5}}}
    tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>
}

// CHECK-LABEL: func @if_else
func @if_else(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>) ->
  (tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>) {
  %graph:3 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
  // expected-remark@above {{Successors: {6}}}
    %island:4 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      %id1 = "tf.Identity"(%arg1) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 1}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      tf_executor.yield %id0, %id0, %id1 :
      // expected-remark@above {{ID: 2}}
        tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2 :
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
      tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>
  }
  return %graph#0, %graph#1, %graph#2 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Predecessors: {5}}}
    tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>
}

// -----

// Tests that the pass tracks control dependencies for variables from a while
// op's output.

// CHECK-LABEL: func @output_of_while_op
func @output_of_while_op(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
  // expected-remark@above {{Successors: {11}}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Successors: {9}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      %while:4 = "tf.While"(%arg2, %id0, %arg1, %arg1) {
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2,3,4}}}
          body = @while_body, cond = @while_cond, is_stateless = false}
        : (tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>) ->
          (tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>,
           tensor<*x!tf.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%while#1) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%while#1) :
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%while#2) :
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {6}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2,3}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
  }
  return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Predecessors: {10}}}
}

// CHECK-LABEL: func @while_body
func @while_body(
  // expected-remark@above {{ID: 7}}
  %pred: tensor<i1>,
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf.resource<tensor<32xf32>>>) ->
  (tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>,
   tensor<*x!tf.resource<tensor<32xf32>>>) {
  %graph:4 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
  // expected-remark@above {{Successors: {6}}}
    %island:5 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      %u0 = "tf._UnknownSideEffectingOp_"() : ()
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2}}}
        -> tensor<*x!tf.resource<tensor<32xf32>>>
      tf_executor.yield %pred, %id0, %arg1, %u0 :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
        tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>,
        tensor<*x!tf.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2, %island#3 :
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
      tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>,
      tensor<*x!tf.resource<tensor<32xf32>>>
  }
  return %graph#0, %graph#1, %graph#2, %graph#3 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Predecessors: {5}}}
    tensor<i1>, tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>,
    tensor<*x!tf.resource<tensor<32xf32>>>
}

// CHECK-LABEL: func @while_cond
func @while_cond(
  // expected-remark@above {{ID: 7}}
  %pred: tensor<i1>,
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<i1> {
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
  // expected-remark@above {{Successors: {6}}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %const = "tf.Const"() { value = dense<0> : tensor<i1> } : () -> tensor<i1>
      // expected-remark@above {{ID: 0}}
      %eq = "tf.Equal"(%pred, %const) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      // expected-remark@above {{ID: 1}}
      tf_executor.yield %eq : tensor<i1>
      // expected-remark@above {{ID: 2}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  return %graph : tensor<i1>
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Predecessors: {5}}}
}

// -----

// Tests that the pass tracks control dependencies based on TF op registry
// statefulness flag, for ops not yet defined in ODS.

// CHECK-LABEL: func @tf_registry_ops
func @tf_registry_ops(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf.string>, %arg1: tensor<!tf.string>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Successors: {7}}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      "tf.PrintV2"(%arg0) { output_stream = "stderr", end = "\n" }
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        : (tensor<!tf.string>) -> ()
      %merge_summary = "tf.MergeSummary"(%arg0, %arg1) { N = 2 }
      // expected-remark@above {{ID: 1}}
        : (tensor<!tf.string>, tensor<!tf.string>) -> (tensor<!tf.string>)
      "tf.PrintV2"(%merge_summary) { output_stream = "stderr", end = "\n" }
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        : (tensor<!tf.string>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Predecessors: {6}}}
}
