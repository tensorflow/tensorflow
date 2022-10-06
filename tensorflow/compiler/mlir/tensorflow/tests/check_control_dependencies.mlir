// RUN: tf-opt -split-input-file -tf-executor-check-control-dependencies -verify-diagnostics %s | FileCheck %s

// expected-error@+1 {{'builtin.module' op not suitable for checking control dependencies}}
module {
  // CHECK-LABEL: func @not_suitable_for_checking
  func.func @not_suitable_for_checking() -> () {
    tf_executor.graph {
      // expected-error@+1 {{functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op}}
      %island = tf_executor.island {
        "tf.OpA"() : () -> ()
        "tf.OpB"() : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    func.return
  }
}

// -----

// Check that we report an unexpected dependency path between two stateless ops.
func.func @two_stateless_ops() -> () {
  tf_executor.graph {
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 0 (source)}}
    %island1 = tf_executor.island {
      "tf.OpA"() {is_stateless = true}: () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 1 (target)}}
    %island2 = tf_executor.island(%island1) {
      "tf.OpB"() {is_stateless = true}: () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Check that we report an unexpected dependency path between a stateless and a
// stateful op.
func.func @source_stateless_target_stateful() -> () {
  tf_executor.graph {
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 0 (source)}}
    %island1 = tf_executor.island {
      "tf.OpA"() {is_stateless = true}: () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 1 (target)}}
    %island2 = tf_executor.island(%island1) {
      "tf.OpB"() {is_stateless = false}: () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Check that we report an unexpected dependency path between a stateful and a
// stateless op.
func.func @source_stateful_target_stateless() -> () {
  tf_executor.graph {
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 0 (source)}}
    %island1 = tf_executor.island {
      "tf.OpA"() {is_stateless = false}: () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 1 (target)}}
    %island2 = tf_executor.island(%island1) {
      "tf.OpB"() {is_stateless = true}: () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Check that we don't report a dependency between two stateful ops.
func.func @two_stateful_ops() -> () {
  tf_executor.graph {
    %island1 = tf_executor.island {
      "tf.OpA"() {is_stateless = false}: () -> ()
      tf_executor.yield
    }
    %island2 = tf_executor.island(%island1) {
      "tf.OpB"() {is_stateless = false}: () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Check that we report a dependency path with intermediate ops between two
// stateful ops that don't access the same resource.
func.func @path_with_intermediate_ops_report(
  %arg0: tensor<!tf_type.resource<tensor<f32>>>,
  %arg1: tensor<!tf_type.resource<tensor<f32>>>,
  %arg2: tensor<f32>) -> () {
  tf_executor.graph {
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 0 (source)}}
    %island1 = tf_executor.island {
      "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 1 (intermediate)}}
    %island2 = tf_executor.island(%island1) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 2 (intermediate)}}
    %island3 = tf_executor.island(%island2) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 3 (target)}}
    %island4 = tf_executor.island(%island3) {
      "tf.AssignVariableOp"(%arg1, %arg2) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// -----

// Check that we don't report a dependency path with intermediate ops between
// two stateful ops that access the same resource.
func.func @path_with_intermediate_ops_no_report(
  %arg0: tensor<!tf_type.resource<tensor<f32>>>,
  %arg1: tensor<f32>) -> () {
  %0 = tf_executor.graph {
    %island1 = tf_executor.island {
      "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    %island2 = tf_executor.island(%island1) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    %island3 = tf_executor.island(%island2) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    %island4:2 = tf_executor.island(%island3) {
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
      tf_executor.yield %read0 : tensor<f32>
    }
    tf_executor.fetch %island4#0 : tensor<f32>
  }
  func.return
}

// -----

// Test a dependency tree with multiple paths, some of them needing reporting
// and some of them not. We expect three reported paths (two of the four
// `AssignVariableOp`s access the same resource variable).
func.func @tree_with_multiple_paths(
  %arg0: tensor<!tf_type.resource<tensor<f32>>>,
  %arg1: tensor<!tf_type.resource<tensor<f32>>>,
  %arg2: tensor<!tf_type.resource<tensor<f32>>>,
  %arg3: tensor<f32>) -> () {
  tf_executor.graph {
    // expected-warning@+2 {{unexpected control dependency path: path 0, node 0 (source)}}
    // expected-warning@+1 {{unexpected control dependency path: path 1, node 0 (source)}}
    %island1 = tf_executor.island {
      "tf.AssignVariableOp"(%arg0, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    // expected-warning@+2 {{unexpected control dependency path: path 0, node 1 (intermediate)}}
    // expected-warning@+1 {{unexpected control dependency path: path 1, node 1 (intermediate)}}
    %island2 = tf_executor.island(%island1) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 2, node 0 (source)}}
    %island3 = tf_executor.island {
      "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 2, node 1 (intermediate)}}
    %island4 = tf_executor.island(%island3) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+3 {{unexpected control dependency path: path 0, node 2 (intermediate)}}
    // expected-warning@+2 {{unexpected control dependency path: path 1, node 2 (intermediate)}}
    // expected-warning@+1 {{unexpected control dependency path: path 2, node 2 (intermediate)}}
    %island5 = tf_executor.island(%island2, %island4) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+2 {{unexpected control dependency path: path 1, node 3 (intermediate)}}
    // expected-warning@+1 {{unexpected control dependency path: path 2, node 3 (intermediate)}}
    %island6 = tf_executor.island(%island5) {
      "tf.NoOp"() : () -> ()
      tf_executor.yield
    }
    // expected-warning@+2 {{unexpected control dependency path: path 1, node 4 (target)}}
    // expected-warning@+1 {{unexpected control dependency path: path 2, node 4 (target)}}
    %island7 = tf_executor.island(%island6) {
      "tf.AssignVariableOp"(%arg2, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    // expected-warning@+1 {{unexpected control dependency path: path 0, node 3 (target)}}
    %island8 = tf_executor.island(%island5) {
      "tf.AssignVariableOp"(%arg1, %arg3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}
