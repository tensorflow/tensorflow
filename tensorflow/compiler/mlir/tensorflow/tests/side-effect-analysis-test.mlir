// RUN: tf-opt -split-input-file -tf-test-side-effect-analysis -verify-diagnostics %s | FileCheck %s

// Tests that the pass tracks control dependencies for reads/writes on the same
// resource.

// CHECK-LABEL: func @non_aliasing_reads_writes
func.func @non_aliasing_reads_writes(
// expected-remark@above {{ID: 13}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 11}}
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Successors: {10}}}
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {1}}}
      "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {6}}}
      %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Successors: {5}}}
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 3}}
      %read2 = "tf.ReadVariableOp"(%var_handle) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%arg1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%arg0, %read2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {7}}}
      %read3 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
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
  func.return %graph : tensor<32xf32>
  // expected-remark@above {{ID: 12}}
  // expected-remark@above {{Sinks: {11}}}
}

// -----

// Tests that the pass tracks control dependencies for reads/writes on the two
// resource handles that refer to the same variable.

// CHECK-LABEL: func @aliasing_reads_writes
func.func @aliasing_reads_writes(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 14}}
  tf_executor.graph {
  // expected-remark@above {{ID: 12}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Successors: {11}}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %vh1_id:2 = "tf.IdentityN"(%vh1, %arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>)
      // expected-remark@above {{ID: 2}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1_id#0, %arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {3}}}
      // expected-remark@above {{Successors: {5,6}}}
      %read1 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
      %read2 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {7}}}
      "tf.AssignVariableOp"(%vh0, %read2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%vh1_id#0, %read1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
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
  func.return
  // expected-remark@above {{ID: 13}}
  // expected-remark@above {{Sinks: {12}}}
}

// -----

// Tests that the pass tracks control dependencies for side-effecting on unknown
// resources.

// CHECK-LABEL: func @unknown_side_effecting_op
func.func @unknown_side_effecting_op(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 14}}
  tf_executor.graph {
  // expected-remark@above {{ID: 12}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 10}}
    // expected-remark@above {{Successors: {11}}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1, %arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Successors: {4}}}
      "tf._UnknownSideEffectingOp_"() : () -> ()
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {2,3}}}
      // expected-remark@above {{Successors: {5,6,7}}}
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {8}}}
      %read2 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {8}}}
      "tf.AssignVariableOp"(%vh0, %read1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {4}}}
      // expected-remark@above {{Successors: {9}}}
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
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
  func.return
  // expected-remark@above {{ID: 13}}
  // expected-remark@above {{Sinks: {12}}}
}

// -----

// Tests that the pass tracks control dependencies for read-only ops on unknown
// resources.

// CHECK-LABEL: func @read_only_unknown_resource
func.func @read_only_unknown_resource(%arg0: tensor<32xf32>) -> () {
// expected-remark@above {{ID: 10}}
  tf_executor.graph {
  // expected-remark@above {{ID: 8}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Successors: {7}}}
      %vh0 = "tf._UnknownSideEffectingOp_"() : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2,3}}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      // expected-remark@above {{ID: 1}}
      %read0 = "tf.ReadVariableOp"(%vh0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {4}}}
      %read1 = "tf.ReadVariableOp"(%vh1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {4}}}
      "tf.AssignVariableOp"(%vh1, %read0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
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
  func.return
  // expected-remark@above {{ID: 9}}
  // expected-remark@above {{Sinks: {8}}}
}

// -----

// Tests that the pass adds control dependencies in nested regions with
// tf_device.replicate

func.func @with_replicate(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg3: tensor<*x!tf_type.resource<tensor<32xf32>>>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
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
          [%arg0, %arg1] as %r0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
          [%arg2, %arg3] as %r1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
          [%u0#0, %u0#1] as %u : tensor<32xf32>)
          {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}} {
        %read0 = "tf.ReadVariableOp"(%r0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
        // expected-remark@above {{ID: 1}}
        "tf.AssignVariableOp"(%r1, %u) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        %read1 = "tf.ReadVariableOp"(%r1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
        tf_device.return
        // expected-remark@above {{ID: 4}}
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
  func.return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Sinks: {10}}}
}

// -----

// Tests that the pass does not add control dependencies for a stateless if op.

// CHECK-LABEL: func @stateless_if_op
func.func @stateless_if_op(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %if = "tf.If"(%arg1, %arg1) {
      // expected-remark@above {{ID: 1}}
          then_branch = @if_then, else_branch = @if_else, is_stateless = true}
        : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// CHECK-LABEL: func @if_then
func.func @if_then(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
  }
  func.return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Sinks: {3}}}
}

// CHECK-LABEL: func @if_else
func.func @if_else(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
  }
  func.return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Sinks: {3}}}
}

// -----

// Tests that the pass does not add control dependencies for a stateless
// IfRegion op.

// CHECK-LABEL: func @stateless_ifregion_op
func.func @stateless_ifregion_op(
  // expected-remark@above {{ID: 18}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 16}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 14}}
    // expected-remark@above {{Successors: {15}}}

      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {12}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>

      %if = "tf.IfRegion"(%arg1) (
      // expected-remark@above {{ID: 11}}
        { // Then region.
          %graph = tf_executor.graph {
          // expected-remark@above {{ID: 4}}
            %island:2 = tf_executor.island {
            // expected-remark@above {{ID: 2}}
              tf_executor.yield %arg1 : tensor<i1>
              // expected-remark@above {{ID: 1}}
            }
            tf_executor.fetch %island#0 : tensor<i1>
            // expected-remark@above {{ID: 3}}
          }
          "tf.Yield"(%graph) : (tensor<i1>) -> ()
          // expected-remark@above {{ID: 5}}
        }, { // Else region
          %graph = tf_executor.graph {
          // expected-remark@above {{ID: 9}}
            %island:2 = tf_executor.island {
            // expected-remark@above {{ID: 7}}
              tf_executor.yield %arg1 : tensor<i1>
              // expected-remark@above {{ID: 6}}
            }
            tf_executor.fetch %island#0 : tensor<i1>
            // expected-remark@above {{ID: 8}}
          }
          "tf.Yield"(%graph) : (tensor<i1>) -> ()
          // expected-remark@above {{ID: 10}}
        }
      ) { is_stateless = true} : (tensor<i1>) -> tensor<i1>

      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 12}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {13}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()

      tf_executor.yield
      // expected-remark@above {{ID: 13}}
      // expected-remark@above {{Predecessors: {12}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 15}}
    // expected-remark@above {{Predecessors: {14}}}
  }
  func.return
  // expected-remark@above {{ID: 17}}
  // expected-remark@above {{Sinks: {16}}}
}

// -----

// Tests that the pass does not add control dependencies a stateless while op.

// CHECK-LABEL: func @stateless_if_op
func.func @stateless_if_op(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %while = "tf.While"(%arg1) {
      // expected-remark@above {{ID: 1}}
          body = @while_body, cond = @while_cond, is_stateless = true}
        : (tensor<i1>) -> tensor<i1>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// CHECK-LABEL: func @while_body
func.func @while_body(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
  }
  func.return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Sinks: {3}}}
}

// CHECK-LABEL: func @while_cond
func.func @while_cond(%arg0: tensor<i1>) -> tensor<i1> {
  // expected-remark@above {{ID: 5}}
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 3}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 1}}
      tf_executor.yield %arg0 : tensor<i1>
      // expected-remark@above {{ID: 0}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 2}}
  }
  func.return %graph : tensor<i1>
  // expected-remark@above {{ID: 4}}
  // expected-remark@above {{Sinks: {3}}}
}

// -----

// Tests that the pass does not add control dependencies a stateless WhileRegion
// op.

// CHECK-LABEL: func @stateless_whileregion_op
func.func @stateless_whileregion_op(
  // expected-remark@above {{ID: 18}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 16}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 14}}
    // expected-remark@above {{Successors: {15}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {12}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>

      %while = "tf.WhileRegion"(%arg1) (
      // expected-remark@above {{ID: 11}}
        {
          ^bb0(%carg: tensor<i1>):
            %graph = tf_executor.graph {
            // expected-remark@above {{ID: 4}}
              %island:2 = tf_executor.island {
              // expected-remark@above {{ID: 2}}
                tf_executor.yield %carg : tensor<i1>
                // expected-remark@above {{ID: 1}}
              }
              tf_executor.fetch %island#0 : tensor<i1>
              // expected-remark@above {{ID: 3}}
            }
            "tf.Yield"(%graph) : (tensor<i1>) -> ()
            // expected-remark@above {{ID: 5}}
        }, {
          ^bb0(%barg: tensor<i1>):
            %graph = tf_executor.graph {
            // expected-remark@above {{ID: 9}}
              %island:2 = tf_executor.island {
              // expected-remark@above {{ID: 7}}
                tf_executor.yield %barg : tensor<i1>
                // expected-remark@above {{ID: 6}}
              }
              tf_executor.fetch %island#0 : tensor<i1>
              // expected-remark@above {{ID: 8}}
            }
            "tf.Yield"(%graph) : (tensor<i1>) -> ()
            // expected-remark@above {{ID: 10}}
        }
      ) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 12}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {13}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 13}}
      // expected-remark@above {{Predecessors: {12}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 15}}
    // expected-remark@above {{Predecessors: {14}}}
  }
  func.return
  // expected-remark@above {{ID: 17}}
  // expected-remark@above {{Sinks: {16}}}
}

// -----

// Tests that the pass tracks control dependencies for variables from an if op's
// output.

// In this test, the resources computed and used are as follows:
// (* = unknown resource id which aliases with everything else)
//   id0 = arg0
//   if-then-branch: [u0,   arg0, arg0]
//   if-else-branch: [arg0, arg0, arg1]
//     => first result is unknown, second and third is passthrough
//   if results    : [*,    arg0, {arg0, arg1}[
//   ID #2: read (unknown)         -> succ {5, 6)
//   ID #3: read (arg0)            -> succ {5}
//   ID #4: read({arg0,arg1})      -> succ {5,6}
//   ID #5: write(arg0)
//   ID #6: write(arg1)

// CHECK-LABEL: func @output_of_if_op
func.func @output_of_if_op(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Successors: {9}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %if:3 = "tf.If"(%arg2, %id0, %arg1) {
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2,3,4}}}
          then_branch = @if_then, else_branch = @if_else, is_stateless = false}
        : (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
          (tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%if#0) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5,6}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%if#1) :
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%if#2) :
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5,6}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2,3,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {2,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
  }
  func.return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Sinks: {10}}}
}

// CHECK-LABEL: func @if_then
func.func @if_then(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
  (tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>) {
  %graph:3 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
    %island:4 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %u0 = "tf._UnknownSideEffectingOp_"() : ()
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 1}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      tf_executor.yield %u0, %id0, %id0 :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
        tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2 :
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
      tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>
  }
  func.return %graph#0, %graph#1, %graph#2 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
    tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>
}

// CHECK-LABEL: func @if_else
func.func @if_else(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
  (tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>) {
  %graph:3 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
    %island:4 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %id1 = "tf.Identity"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 1}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      tf_executor.yield %id0, %id0, %id1 :
      // expected-remark@above {{ID: 2}}
        tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2 :
    // expected-remark@above {{ID: 4}}
      tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>
  }
  func.return %graph#0, %graph#1, %graph#2 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
    tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>
}

// -----

// Tests that the pass tracks control dependencies for variables from an
// IfRegion op's output.

// CHECK-LABEL: func @output_of_ifregion_op
func.func @output_of_ifregion_op(
  // expected-remark@above {{ID: 26}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 24}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 22}}
    // expected-remark@above {{Successors: {23}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %if:3 = "tf.IfRegion"(%arg2) (
      // expected-remark@above {{ID: 15}}
      // expected-remark@above {{Successors: {16,17,18}}}
        {
          %graph:3 = tf_executor.graph {
          // expected-remark@above {{ID: 6}}
            %island:4 = tf_executor.island {
            // expected-remark@above {{ID: 4}}
            // expected-remark@above {{Successors: {5}}}
              %u0 = "tf._UnknownSideEffectingOp_"() : ()
              // expected-remark@above {{ID: 1}}
              // expected-remark@above {{Successors: {3}}}
                -> tensor<*x!tf_type.resource<tensor<32xf32>>>
              %iid0 = "tf.Identity"(%id0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
              // expected-remark@above {{ID: 2}}
                -> tensor<*x!tf_type.resource<tensor<32xf32>>>
              tf_executor.yield %u0, %iid0, %iid0 :
              // expected-remark@above {{ID: 3}}
              // expected-remark@above {{Predecessors: {1}}}
                tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>
            }
            tf_executor.fetch %island#0, %island#1, %island#2 :
            // expected-remark@above {{ID: 5}}
            // expected-remark@above {{Predecessors: {4}}}
              tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>
          }
          "tf.Yield"(%graph#0, %graph#1, %graph#2) :
          // expected-remark@above {{ID: 7}}
            (tensor<*x!tf_type.resource<tensor<32xf32>>>,
            tensor<*x!tf_type.resource<tensor<32xf32>>>,
            tensor<*x!tf_type.resource<tensor<32xf32>>>) -> ()
        },
        {
          %graph:3 = tf_executor.graph {
          // expected-remark@above {{ID: 13}}
            %island:4 = tf_executor.island {
            // expected-remark@above {{ID: 11}}
              %iid0 = "tf.Identity"(%id0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
              // expected-remark@above {{ID: 8}}
                -> tensor<*x!tf_type.resource<tensor<32xf32>>>
              %iid1 = "tf.Identity"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
              // expected-remark@above {{ID: 9}}
                -> tensor<*x!tf_type.resource<tensor<32xf32>>>
              tf_executor.yield %iid0, %iid0, %iid1 :
              // expected-remark@above {{ID: 10}}
                tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>
            }
            tf_executor.fetch %island#0, %island#1, %island#2 :
            // expected-remark@above {{ID: 12}}
              tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>
          }
          "tf.Yield"(%graph#0, %graph#1, %graph#2) :
          // expected-remark@above {{ID: 14}}
            (tensor<*x!tf_type.resource<tensor<32xf32>>>,
            tensor<*x!tf_type.resource<tensor<32xf32>>>,
            tensor<*x!tf_type.resource<tensor<32xf32>>>) -> ()
        }) { is_stateless = false}
        : (tensor<i1>) ->
          (tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%if#0) :
      // expected-remark@above {{ID: 16}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {19,20}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%if#1) :
      // expected-remark@above {{ID: 17}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {19}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%if#2) :
      // expected-remark@above {{ID: 18}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {19,20}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 19}}
      // expected-remark@above {{Predecessors: {16,17,18}}}
      // expected-remark@above {{Successors: {21}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 20}}
      // expected-remark@above {{Predecessors: {16,18}}}
      // expected-remark@above {{Successors: {21}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 21}}
      // expected-remark@above {{Predecessors: {19,20}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 23}}
    // expected-remark@above {{Predecessors: {22}}}
  }
  func.return
  // expected-remark@above {{ID: 25}}
  // expected-remark@above {{Sinks: {24}}}
}

// -----

// Tests that the pass tracks control dependencies for variables from a while
// op's output.

// Here:
//   id0 = arg0
//   while-inputs = (id0/arg0, arg1, arg1)
//   while body pass through first and second arg, not last one
//   while-results = (arg0, arg1, Unknown)
//   #ID 2: read(arg0)      -> succ{5}
//   #ID 3: read(arg1)      -> succ{6}
//   #ID 4: read(unknown)   -> succ{5,6}
//   #ID 5 : write(arg0)
//   #ID 6 : write(arg1)


// CHECK-LABEL: func @output_of_while_op
func.func @output_of_while_op(
  // expected-remark@above {{ID: 12}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 10}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Successors: {9}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %while:4 = "tf.While"(%arg2, %id0, %arg1, %arg1) {
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2,3,4}}}
          body = @while_body, cond = @while_cond, is_stateless = false}
        : (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
          (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%while#1) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%while#2) :
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {6}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%while#3) :
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {1}}}
      // expected-remark@above {{Successors: {5,6}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 5}}
      // expected-remark@above {{Predecessors: {2,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 6}}
      // expected-remark@above {{Predecessors: {3,4}}}
      // expected-remark@above {{Successors: {7}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 7}}
      // expected-remark@above {{Predecessors: {5,6}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 9}}
    // expected-remark@above {{Predecessors: {8}}}
  }
  func.return
  // expected-remark@above {{ID: 11}}
  // expected-remark@above {{Sinks: {10}}}
}

// CHECK-LABEL: func @while_body
func.func @while_body(
  // expected-remark@above {{ID: 7}}
  %pred: tensor<i1>,
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
  (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>,
   tensor<*x!tf_type.resource<tensor<32xf32>>>) {
  %graph:4 = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
    %island:5 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %u0 = "tf._UnknownSideEffectingOp_"() : ()
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {2}}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      tf_executor.yield %pred, %id0, %arg1, %u0 :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {1}}}
        tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>,
        tensor<*x!tf_type.resource<tensor<32xf32>>>
    }
    tf_executor.fetch %island#0, %island#1, %island#2, %island#3 :
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
      tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>,
      tensor<*x!tf_type.resource<tensor<32xf32>>>
  }
  func.return %graph#0, %graph#1, %graph#2, %graph#3 :
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
    tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>,
    tensor<*x!tf_type.resource<tensor<32xf32>>>
}

// CHECK-LABEL: func @while_cond
func.func @while_cond(
  // expected-remark@above {{ID: 7}}
  %pred: tensor<i1>,
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<i1> {
  %graph = tf_executor.graph {
  // expected-remark@above {{ID: 5}}
    %island:2 = tf_executor.island {
    // expected-remark@above {{ID: 3}}
      %const = "tf.Const"() { value = dense<0> : tensor<i1> } : () -> tensor<i1>
      // expected-remark@above {{ID: 0}}
      %eq = "tf.Equal"(%pred, %const) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      // expected-remark@above {{ID: 1}}
      tf_executor.yield %eq : tensor<i1>
      // expected-remark@above {{ID: 2}}
    }
    tf_executor.fetch %island#0 : tensor<i1>
    // expected-remark@above {{ID: 4}}
  }
  func.return %graph : tensor<i1>
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that the pass tracks control dependencies for variables from a
// WhileRegion op's output.

// CHECK-LABEL: func @output_of_whileregion_op
func.func @output_of_whileregion_op(
  // expected-remark@above {{ID: 26}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<i1>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 24}}
    // CHECK: tf_executor.island
    %island = tf_executor.island {
    // expected-remark@above {{ID: 22}}
    // expected-remark@above {{Successors: {23}}}
      %id0 = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
      // expected-remark@above {{ID: 0}}
        -> tensor<*x!tf_type.resource<tensor<32xf32>>>
      %while:4 = "tf.WhileRegion"(%arg2, %id0, %arg1, %arg1) (
      // expected-remark@above {{ID: 15}}
      // expected-remark@above {{Successors: {16,17,18}}}
        {
          ^bb0(%pred: tensor<i1>,
               %carg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
               %carg2: tensor<*x!tf_type.resource<tensor<32xf32>>>,
               %carg3: tensor<*x!tf_type.resource<tensor<32xf32>>>):
            %graph = tf_executor.graph {
            // expected-remark@above {{ID: 6}}
              %island:2 = tf_executor.island {
              // expected-remark@above {{ID: 4}}
                %const = "tf.Const"() { value = dense<0> : tensor<i1> } : () -> tensor<i1>
                // expected-remark@above {{ID: 1}}
                %eq = "tf.Equal"(%pred, %const) : (tensor<i1>, tensor<i1>) -> tensor<i1>
                // expected-remark@above {{ID: 2}}
                tf_executor.yield %eq : tensor<i1>
                // expected-remark@above {{ID: 3}}
              }
              tf_executor.fetch %island#0 : tensor<i1>
              // expected-remark@above {{ID: 5}}
            }
            "tf.Yield"(%graph) : (tensor<i1>) -> ()
            // expected-remark@above {{ID: 7}}
        },
        {
          ^bb0(%pred: tensor<i1>,
               %barg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
               %barg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
               %barg2: tensor<*x!tf_type.resource<tensor<32xf32>>>):
             %graph:4 = tf_executor.graph {
            // expected-remark@above {{ID: 13}}
              %island:5 = tf_executor.island {
              // expected-remark@above {{ID: 11}}
              // expected-remark@above {{Successors: {12}}}
                %iid0 = "tf.Identity"(%barg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>)
                // expected-remark@above {{ID: 8}}
                  -> tensor<*x!tf_type.resource<tensor<32xf32>>>
                %u0 = "tf._UnknownSideEffectingOp_"() : ()
                // expected-remark@above {{ID: 9}}
                // expected-remark@above {{Successors: {10}}}
                  -> tensor<*x!tf_type.resource<tensor<32xf32>>>
                tf_executor.yield %pred, %iid0, %barg1, %u0 :
                // expected-remark@above {{ID: 10}}
                // expected-remark@above {{Predecessors: {9}}}
                  tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
                  tensor<*x!tf_type.resource<tensor<32xf32>>>,
                  tensor<*x!tf_type.resource<tensor<32xf32>>>
              }
              tf_executor.fetch %island#0, %island#1, %island#2, %island#3 :
              // expected-remark@above {{ID: 12}}
              // expected-remark@above {{Predecessors: {11}}}
                tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>,
                tensor<*x!tf_type.resource<tensor<32xf32>>>
            }
            "tf.Yield"(%graph#0, %graph#1, %graph#2, %graph#3) :
            // expected-remark@above {{ID: 14}}
              (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>,
              tensor<*x!tf_type.resource<tensor<32xf32>>>) -> ()
        }
      ) {is_stateless = false}
        : (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>) ->
          (tensor<i1>, tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>,
           tensor<*x!tf_type.resource<tensor<32xf32>>>)
      %r0 = "tf.ReadVariableOp"(%while#1) :
      // expected-remark@above {{ID: 16}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {19}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%while#2) :
      // expected-remark@above {{ID: 17}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {20}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%while#3) :
      // expected-remark@above {{ID: 18}}
      // expected-remark@above {{Predecessors: {15}}}
      // expected-remark@above {{Successors: {19,20}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %r0) :
      // expected-remark@above {{ID: 19}}
      // expected-remark@above {{Predecessors: {16,18}}}
      // expected-remark@above {{Successors: {21}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 20}}
      // expected-remark@above {{Predecessors: {17,18}}}
      // expected-remark@above {{Successors: {21}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 21}}
      // expected-remark@above {{Predecessors: {19,20}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 23}}
    // expected-remark@above {{Predecessors: {22}}}
  }
  func.return
  // expected-remark@above {{ID: 25}}
  // expected-remark@above {{Sinks: {24}}}
}

// -----

// Tests that the pass tracks control dependencies based on TF op registry
// statefulness flag, for ops not yet defined in ODS.

// CHECK-LABEL: func @tf_registry_ops
func.func @tf_registry_ops(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>, %arg1: tensor<!tf_type.string>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Successors: {5}}}
      "tf.PrintV2"(%arg0) { output_stream = "stderr", end = "\n" }
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {2}}}
        : (tensor<!tf_type.string>) -> ()
      %merge_summary = "tf.MergeSummary"(%arg0, %arg1) { N = 2 }
      // expected-remark@above {{ID: 1}}
        : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<!tf_type.string>)
      "tf.PrintV2"(%merge_summary) { output_stream = "stderr", end = "\n" }
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Predecessors: {0}}}
      // expected-remark@above {{Successors: {3}}}
        : (tensor<!tf_type.string>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that the pass tracks control dependencies for resource arguments with
// aliasing table (unique IDs).

// CHECK-LABEL: func @arguments_with_unique_ids
func.func @arguments_with_unique_ids(
  // expected-remark@above {{ID: 9}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf._resource_arg_unique_id = 0 : i64},
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf._resource_arg_unique_id = 0 : i64},
  %arg2: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf._resource_arg_unique_id = 33 : i64}) {
  tf_executor.graph {
  // expected-remark@above {{ID: 7}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Successors: {6}}}
      %r0 = "tf.ReadVariableOp"(%arg0) :
      // expected-remark@above {{ID: 0}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r1 = "tf.ReadVariableOp"(%arg1) :
      // expected-remark@above {{ID: 1}}
      // expected-remark@above {{Successors: {3}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %r2 = "tf.ReadVariableOp"(%arg2) :
      // expected-remark@above {{ID: 2}}
      // expected-remark@above {{Successors: {4}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg1, %r0) :
      // expected-remark@above {{ID: 3}}
      // expected-remark@above {{Predecessors: {0,1}}}
      // expected-remark@above {{Successors: {4}}}
        (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      tf_executor.yield
      // expected-remark@above {{ID: 4}}
      // expected-remark@above {{Predecessors: {2,3}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests interplay of value-based side-effects for non-resource values and
// unknown side effects.
func.func @value_based_side_effect_non_resource_to_unknown(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        "tf._InternalTestNonResourceValueSideEffects_"(%arg0) : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests interplay of value-based side-effects for non-resource values and
// other known side effects.
func.func @value_based_side_effect_non_resource_to_known(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        %0 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf._InternalTestNonResourceValueSideEffects_"(%arg0) : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {3}}}
        %1 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {1,2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that the analysis correctly handles a sequence of ops using the
// `DatasetIterator` resource.
func.func @dataset_op_sequence(
  // expected-remark@above {{ID: 9}}
  %arg0: tensor<!tf_type.variant>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 7}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 5}}
        // expected-remark@above {{Successors: {6}}}
        %handle, %deleter = "tf.AnonymousIteratorV2"() {_class = ["loc:@GeneratorDataset_2"], device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string]} : () -> (tensor<!tf_type.resource>, tensor<!tf_type.variant>)
        // expected-remark@above {{ID: 0}}
        "tf.MakeIterator"(%arg0, %handle) {_class = ["loc:@GeneratorDataset_2"], device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.variant>, tensor<!tf_type.resource>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
         %0 = "tf.IteratorGetNext"(%handle) {_class = ["loc:@GeneratorDataset_2"], device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource>) -> tensor<!tf_type.string>
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
        // expected-remark@above {{Successors: {3}}}
        "tf.DeleteIterator"(%handle, %deleter) {device = ""} : (tensor<!tf_type.resource>, tensor<!tf_type.variant>) -> ()
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
        // expected-remark@above {{Successors: {4}}}
        tf_executor.yield
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Predecessors: {3}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests `tf.GeneratorDataset` with surrounding ops with unknown side-effects.
func.func @generator_dataset_with_unknown_side_effect_ops(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        %0 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that two resources allocated with identical non-empty `shared_name`
// attributes are dependent. That means, `%handle1` and `%handle2` point to
// the same resources and the second `InitializeTableV2` op depends on the
// first one.
func.func @resources_allocated_with_same_nonempty_shared_name(
  // expected-remark@above {{ID: 9}}
  %key: tensor<!tf_type.string>,
  %value: tensor<i64>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 7}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Successors: {6}}}
        %handle1 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "some_name", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
        // expected-remark@above {{ID: 0}}
        %handle2 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "some_name", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
        // expected-remark@above {{ID: 1}}
        "tf.InitializeTableV2"(%handle1, %key, %value) : (tensor<!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        "tf.InitializeTableV2"(%handle2, %key, %value) : (tensor<!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
        // expected-remark@above {{Successors: {4}}}
        tf_executor.yield
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Predecessors: {3}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests two side-effecting ops operating on resources passed as function
// parameters. The expectation is that the ops are treated as independent (as
// no `tf._resource_arg_unique_id` attributes are present).
func.func @side_effecting_ops_with_different_resources(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.resource>,
  %arg1: tensor<!tf_type.resource>,
  %arg2: tensor<f32>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Successors: {4}}}
        %0 = "tf.StackPushV2"(%arg0, %arg2) {device = "", swap_memory = false} : (tensor<!tf_type.resource>, tensor<f32>) -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        %1 = "tf.StackPushV2"(%arg1, %arg2) {device = "", swap_memory = false} : (tensor<!tf_type.resource>, tensor<f32>) -> tensor<f32>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0,1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests two side-effecting ops operating on resources that are allocated in the
// same function. The expectation is that the ops are treated as independent
// (as the involved resource allocators have the `UniqueResourceAllocation`
// trait).
func.func @side_effecting_ops_with_different_resources_and_allocations(
  // expected-remark@above {{ID: 9}}
  %arg0: tensor<i32>,
  %arg1: tensor<f32>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 7}}
    %island = tf_executor.island {
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Successors: {6}}}
        %stack_handle1 = "tf.StackV2"(%arg0) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf_type.resource>
        // expected-remark@above {{ID: 0}}
        %stack_handle2 = "tf.StackV2"(%arg0) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf_type.resource>
        // expected-remark@above {{ID: 1}}
        %0 = "tf.StackPushV2"(%stack_handle1, %arg1) {device = "", swap_memory = false} : (tensor<!tf_type.resource>, tensor<f32>) -> tensor<f32>
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {4}}}
        %1 = "tf.StackPushV2"(%stack_handle2, %arg1) {device = "", swap_memory = false} : (tensor<!tf_type.resource>, tensor<f32>) -> tensor<f32>
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        tf_executor.yield
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Predecessors: {2,3}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests that we create a dependency for op instances with
// `TPUEmbeddingSideEffect` with same device ordinal.
func.func @embedding_effect_same_device(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0) {table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0) {table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we treat different op instances with `TPUEmbeddingSideEffect` as
// independent if they have different device ordinals.
func.func @embedding_effect_different_devices(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0) {table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0) {table_ids = [1, 2], device_ordinal = 2} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0,1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we create dependencies between ops with `TPUEmbeddingSideEffect`
// and unknown side-effecting ops.
func.func @mixed_embedding_and_unknown_effects(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we don't create dependencies between ops `EnqueueTPUEmbedding`
// ops and other embedding ops that don't have a device ordinal.
func.func @mixed_embedding_and_unknown_effects(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<8xf32>,
  %arg2: tensor<8xf32>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {3}}}
        "tf.LoadTPUEmbeddingAdagradParameters"(%arg1, %arg2) {config = "", num_shards = 1 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "table1"} : (tensor<8xf32>, tensor<8xf32>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {3}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2], device_ordinal = 2} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {0,1,2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we create a dependency between two ops with the same op-based
// write effect.
func.func @same_op_based_write_effect(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        %0 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        %1 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we treat ops with different op-based side effects as independent.
func.func @different_op_based_side_effects(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {3}}}
        %0 = "tf.GeneratorDataset"(%arg0, %arg0, %arg0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {3}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2], device_ordinal = 5} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {0,1,2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we don't create dependencies between ops with different op-based
// and value-based side effects.
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
func.func @mixed_op_based_value_based_side_effects(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: !tf_res,
  %arg2: tensor<f32>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf.AssignVariableOp"(%arg1, %arg2) : (!tf_res, tensor<f32>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {3}}}
        "tf.ReadVariableOp"(%arg1) : (!tf_res) -> tensor<f32>
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {1,2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we create dependencies between `_XlaRecvAtHostV2` ops with equal
// keys.
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
func.func @recv_equal_keys(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        %0 = "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        %const = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
        // expected-remark@above {{ID: 1}}
        %1 = "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we create dependencies between `_XlaSendFromHostV2` ops with equal
// keys.
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
func.func @send_equal_keys(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        %const = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        "tf._XlaSendFromHostV2"(%const, %arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_retvals"} : (tensor<f32>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
        "tf._XlaSendFromHostV2"(%const, %arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_retvals"} : (tensor<f32>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}

  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we don't create dependencies between `_XlaRecvAtHostV2` ops with
// different keys (corresponding to different resources).
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
func.func @recv_different_keys(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        %0 = "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        %const = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
        // expected-remark@above {{ID: 1}}
        %1 = "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_1_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 2}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}
// -----

// Tests that we don't create dependencies between `_XlaSendFromHostV2` ops with
// different keys (corresponding to different resources).
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
func.func @send_different_keys(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        %const = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        "tf._XlaSendFromHostV2"(%const, %arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_0_retvals"} : (tensor<f32>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 1}}
        "tf._XlaSendFromHostV2"(%const, %arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_1_retvals"} : (tensor<f32>, tensor<!tf_type.string>, tensor<i64>) -> ()
        // expected-remark@above {{ID: 2}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we create a dependency between ops with `TF_TPUExecuteSideEffect`.
func.func @tpu_execute_effect(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        "tf.TPUExecute"(%arg0, %arg0) : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        "tf.TPUExecute"(%arg1, %arg1) : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we don't create dependencies between TPU compile ops.
func.func @tpu_compile_ops(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        %0:2 = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
        // expected-remark@above {{ID: 0}}
        %1:2 = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
        // expected-remark@above {{ID: 1}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that `_TPUDeviceOrdinalPlaceholder` is side-effect-free.
func.func @device_ordinal_placeholder_side_effect_free(
  // expected-remark@above {{ID: 7}}
  ) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        "tf._TPUDeviceOrdinalPlaceholder"() : () -> tensor<i64>
        // expected-remark@above {{ID: 0}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we don't create dependencies from or to ops with `TF_MustExecute`
// trait.
func.func @must_execute_ops(
  // expected-remark@above {{ID: 7}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<!tf_type.string>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf._InternalTestMustExecuteTrait_"() : () -> ()
        // expected-remark@above {{ID: 1}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}

// -----

// Tests that we don't create dependencies between ops with resources that only
// have self-dependency (like `_XlaRecvAtHostV2`) to or from ops with unknown
// resources.
func.func @no_unknown_side_effect_dependencies(
  // expected-remark@above {{ID: 8}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 6}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {5}}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_1_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 1}}
        "tf._UnknownSideEffectingOp_"() : () -> ()
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {3}}}
        tf_executor.yield
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Predecessors: {2}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {4}}}
  }
  func.return
  // expected-remark@above {{ID: 7}}
  // expected-remark@above {{Sinks: {6}}}
}

// -----

// Tests that we create dependencies from ops with resources that only
// have self-dependency (the island containing `_XlaRecvAtHostV2`) to `Fetch`.
func.func @fetch_dependencies(
  // expected-remark@above {{ID: 6}}
  %arg0: tensor<!tf_type.string>,
  %arg1: tensor<i64>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 4}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Successors: {3}}}
        "tf._XlaRecvAtHostV2"(%arg0, %arg1) {_xla_has_host_transfer = true, key = "host_compute_channel_1_args"} : (tensor<!tf_type.string>, tensor<i64>) -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        tf_executor.yield
        // expected-remark@above {{ID: 1}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 3}}
    // expected-remark@above {{Predecessors: {2}}}
  }
  func.return
  // expected-remark@above {{ID: 5}}
  // expected-remark@above {{Sinks: {4}}}
}

// -----

// Tests that we don't create dependencies between islands with one stateless op
// each.
func.func @single_stateless_op_islands() {
  // expected-remark@above {{ID: 9}}
  tf_executor.graph {
  // expected-remark@above {{ID: 7}}
    %island1 = tf_executor.island {
        // expected-remark@above {{ID: 2}}
        "tf.A"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 0}}
        tf_executor.yield
        // expected-remark@above {{ID: 1}}
    }
    %island2 = tf_executor.island {
        // expected-remark@above {{ID: 5}}
        "tf.B"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 3}}
        tf_executor.yield
        // expected-remark@above {{ID: 4}}
    }
    tf_executor.fetch %island1, %island2 : !tf_executor.control, !tf_executor.control
    // expected-remark@above {{ID: 6}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests that we create dependencies between islands with one stateful op each.
func.func @single_stateful_op_islands() {
  // expected-remark@above {{ID: 9}}
  tf_executor.graph {
  // expected-remark@above {{ID: 7}}
    %island1 = tf_executor.island {
    // expected-remark@above {{ID: 2}}
    // expected-remark@above {{Successors: {5}}}
        "tf.A"() {is_stateless=false} : () -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        tf_executor.yield
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
    }
    %island2 = tf_executor.island {
    // expected-remark@above {{ID: 5}}
    // expected-remark@above {{Predecessors: {2}}}
    // expected-remark@above {{Successors: {6}}}
        "tf.B"() {is_stateless=false} : () -> ()
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        tf_executor.yield
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Predecessors: {3}}}
    }
    tf_executor.fetch %island1, %island2 : !tf_executor.control, !tf_executor.control
    // expected-remark@above {{ID: 6}}
    // expected-remark@above {{Predecessors: {5}}}
  }
  func.return
  // expected-remark@above {{ID: 8}}
  // expected-remark@above {{Sinks: {7}}}
}

// -----

// Tests that we don't create dependencies between islands with multiple
// stateless ops each.
func.func @multi_stateless_op_islands() {
  // expected-remark@above {{ID: 11}}
  tf_executor.graph {
  // expected-remark@above {{ID: 9}}
    %island1 = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        "tf.A"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 0}}
        "tf.B"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 1}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
    }
    %island2 = tf_executor.island {
        // expected-remark@above {{ID: 7}}
        "tf.C"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 4}}
        "tf.D"() {is_stateless=true} : () -> ()
        // expected-remark@above {{ID: 5}}
        tf_executor.yield
        // expected-remark@above {{ID: 6}}
    }
    tf_executor.fetch %island1, %island2 : !tf_executor.control, !tf_executor.control
    // expected-remark@above {{ID: 8}}
  }
  func.return
  // expected-remark@above {{ID: 10}}
  // expected-remark@above {{Sinks: {9}}}
}

// -----

// Tests that we create dependencies between islands that write to the same
// resource variable and also have different op-based effects.
func.func @nontrivial_multi_op_islands(
  // expected-remark@above {{ID: 11}}
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<32xf32>,
  %arg2: tensor<!tf_type.string>) {
  tf_executor.graph {
  // expected-remark@above {{ID: 9}}
    %island1 = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {7,8}}}
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg2) {table_ids = [1, 2], device_ordinal = 1} : (tensor<!tf_type.string>) -> ()
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {2}}}
        "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {0,1}}}
    }
    %island2 = tf_executor.island {
        // expected-remark@above {{ID: 7}}
        // expected-remark@above {{Successors: {8}}}
        // expected-remark@above {{Predecessors: {3}}}
        %0 = "tf.GeneratorDataset"(%arg2, %arg2, %arg2) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", finalize_func = @__func_a, init_func = @__func_b, next_func = @__func_c, next_func.experimental_ints_on_device = true, operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], metadata = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.variant>
        // expected-remark@above {{ID: 4}}
        // expected-remark@above {{Successors: {6}}}
        "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
        // expected-remark@above {{ID: 5}}
        // expected-remark@above {{Successors: {6}}}
        tf_executor.yield
        // expected-remark@above {{ID: 6}}
        // expected-remark@above {{Predecessors: {4,5}}}
    }
    tf_executor.fetch %island1, %island2 : !tf_executor.control, !tf_executor.control
    // expected-remark@above {{ID: 8}}
    // expected-remark@above {{Predecessors: {3,7}}}
  }
  func.return
  // expected-remark@above {{ID: 10}}
  // expected-remark@above {{Sinks: {9}}}
}

// -----

// Tests that we create dependencies between `CollectiveReduceV2` ops
// (TF_CollectiveReduceOrderingEffect).
func.func @collective_reduce_ordering_effect(
  // expected-remark@above {{ID: 7}}
  %input: tensor<f32>,
  %group_key: tensor<i32>,
  %group_size: tensor<i32>,
  %instance_key: tensor<i32>) {
  tf_executor.graph {
    // expected-remark@above {{ID: 5}}
    %island = tf_executor.island {
        // expected-remark@above {{ID: 3}}
        // expected-remark@above {{Successors: {4}}}
        %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
        // expected-remark@above {{ID: 0}}
        // expected-remark@above {{Successors: {1}}}
        %1 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Mul", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
        // expected-remark@above {{ID: 1}}
        // expected-remark@above {{Predecessors: {0}}}
        // expected-remark@above {{Successors: {2}}}
        tf_executor.yield
        // expected-remark@above {{ID: 2}}
        // expected-remark@above {{Predecessors: {1}}}
    }
    tf_executor.fetch %island : !tf_executor.control
    // expected-remark@above {{ID: 4}}
    // expected-remark@above {{Predecessors: {3}}}
  }
  func.return
  // expected-remark@above {{ID: 6}}
  // expected-remark@above {{Sinks: {5}}}
}
