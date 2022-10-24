// RUN: tf-opt -split-input-file -verify-diagnostics -tf-resource-device-inference %s | FileCheck %s

!tf_res = tensor<*x!tf_type.resource<tensor<32xf32>>>

// Tests that the pass can correctly propagate device attributes inside the same
// function.

// CHECK-LABEL: func @propagate_in_function
func.func @propagate_in_function(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: !tf_res {tf.device = "/TPU:1"}) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/CPU:0"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id1 = "tf.Identity"(%id0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/CPU:0"}
      %id2 = "tf.Identity"(%var_handle) : (!tf_res)
        -> !tf_res
      %read = "tf.ReadVariableOp"(%id2) : (!tf_res) -> tensor<32xf32>
      %id3 = "tf.Identity"(%read) : (tensor<32xf32>) -> tensor<32xf32>
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// -----
!tf_res = tensor<*x!tf_type.resource<tensor<32xf32>>>

// Tesets that the pass can propagate through tf.If's branches.

// CHECK-LABEL: func @propagate_if_op
func.func @propagate_if_op(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i1>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.If"
      "tf.If"(%arg1, %id0, %var_handle) {
          then_branch = @if_then,
          else_branch = @if_else,
          is_stateless = false}
        : (tensor<i1>, !tf_res,
           !tf_res) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @if_then
func.func @if_then(
  %arg0: !tf_res,
  %arg1: !tf_res) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:1"}
      %id1 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @if_else
func.func @if_else(
  %arg0: !tf_res,
  %arg1: !tf_res) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}


// -----
!tf_res = tensor<*x!tf_type.resource<tensor<32xf32>>>

// Tesets that the pass can propagate through tf.While's branches.
// CHECK-LABEL: func @propagate_while_op
func.func @propagate_while_op(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.While"
      "tf.While"(%arg1, %id0, %var_handle) {
          body = @while_body,
          cond = @while_cond, is_stateless = false}
        : (tensor<i32>, !tf_res,
           !tf_res) ->
          (tensor<i32>, !tf_res,
           !tf_res)
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @while_body
func.func @while_body(
  %arg0: tensor<i32>,
  %arg1: !tf_res,
  %arg2: !tf_res) ->
  (tensor<i32>, !tf_res,
   !tf_res) {
  %graph:3 = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:4 = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:1"}
      %id1 = "tf.Identity"(%arg2) : (!tf_res)
        -> !tf_res
      tf_executor.yield %arg0, %id0, %id1
        : tensor<i32>, !tf_res,
          !tf_res
    }
    tf_executor.fetch %island#0, %island#1, %island#2
      : tensor<i32>, !tf_res,
        !tf_res
  }
  func.return %graph#0, %graph#1, %graph#2
     : tensor<i32>, !tf_res,
       !tf_res
}

// CHECK-LABEL: func @while_cond
func.func @while_cond(
  %arg0: tensor<i32>,
  %arg1: !tf_res,
  %arg2: !tf_res) -> tensor<32xf32> {
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      %read = "tf.ReadVariableOp"(%id0)
        : (!tf_res) -> tensor<32xf32>
      tf_executor.yield %read : tensor<32xf32>
    }
    tf_executor.fetch %island#0 : tensor<32xf32>
  }
  func.return %graph : tensor<32xf32>
}

// -----
!tf_res = tensor<*x!tf_type.resource<tensor<32xf32>>>

// Tesets that the pass reports error on conflicting assignments from multiple
// callers.

func.func @error_on_conflict_multiple_callers(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i1>) {
  tf_executor.graph {
    %island = tf_executor.island {
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"}
        : () -> !tf_res
      "tf.If"(%arg1, %id0, %var_handle) {
          then_branch = @if_then_and_else,
          else_branch = @if_then_and_else, is_stateless = false}
        : (tensor<i1>, !tf_res,
           !tf_res) -> ()
      "tf.If"(%arg1, %var_handle, %id0) {
      // expected-error@above {{Conflicting device assignment for resource}}
          then_branch = @if_then_and_else,
          else_branch = @if_then_and_else,
          is_stateless = false}
        : (tensor<i1>, !tf_res,
           !tf_res) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

func.func @if_then_and_else(
  %arg0: !tf_res,
  %arg1: !tf_res) {
  tf_executor.graph {
    %island = tf_executor.island {
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      %id1 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// -----

// Test that the pass can propagate through calls
!tf_res = tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @test_function
// CHECK-SAME: {tf.device = "/TPU:0"}
func.func @test_function(%arg0: !tf_res) {
  // CHECK: "tf.Identity"
  // CHECK-SAME: {device = "/TPU:0"}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  %read = "tf.ReadVariableOp"(%id0) : (!tf_res) -> tensor<32xf32>
  %cst = arith.constant dense<3.0> : tensor<32xf32>
  %add = "tf.AddV2"(%read, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  "tf.AssignVariableOp"(%arg0, %add) : (!tf_res, tensor<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: func @propagate_through_calls
func.func @propagate_through_calls(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: !tf_res {tf.device = "/TPU:1"}) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/CPU:0"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id1 = "tf.Identity"(%id0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/CPU:0"}
      %id2 = "tf.Identity"(%var_handle) : (!tf_res)
        -> !tf_res
      %read = "tf.ReadVariableOp"(%id2) : (!tf_res) -> tensor<32xf32>
      %id3 = "tf.Identity"(%read) : (tensor<32xf32>) -> tensor<32xf32>
      func.call @test_function(%id1) : (!tf_res) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// Test propagation through IfRegion (with non-inlined calls)
// CHECK-LABEL: func @propagate_if_region
func.func @propagate_if_region(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i1>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.IfRegion"
      "tf.IfRegion"(%arg1) ({
          func.call @ifregion_then(%id0, %var_handle) : (!tf_res, !tf_res) -> ()
          "tf.Yield"() : () -> ()
        }, {
          func.call @ifregion_else(%id0, %var_handle) : (!tf_res, !tf_res) -> ()
          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @ifregion_then
// CHECK-SAME: (%arg0: {{.+}} {tf.device = "/TPU:0"}, %arg1: {{.+}} {tf.device = "/TPU:1"}
func.func @ifregion_then(
  %arg0: !tf_res,
  %arg1: !tf_res) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:1"}
      %id1 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @ifregion_else
// CHECK-SAME: (%arg0: {{.+}} {tf.device = "/TPU:0"}, %arg1: {{.+}} {tf.device = "/TPU:1"}
func.func @ifregion_else(
  %arg0: !tf_res,
  %arg1: !tf_res) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:1"}
      %id1 = "tf.Identity"(%arg1) : (!tf_res)
        -> !tf_res
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// Test progagation through IfRegion (inlined calls)
// CHECK-LABEL: func @propagate_if_region_inlined
func.func @propagate_if_region_inlined(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i1>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res)
        -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"}
        : () -> !tf_res
      // CHECK-NEXT: "tf.IfRegion"
      "tf.IfRegion"(%arg1) ({
          tf_executor.graph {
             // CHECK: tf_executor.island
             %island = tf_executor.island {
               // CHECK-NEXT: "tf.Identity"
               // CHECK-SAME: {device = "/TPU:0"}
               %id1 = "tf.Identity"(%id0) : (!tf_res) -> !tf_res
               // CHECK-NEXT: "tf.Identity"
               // CHECK-SAME: {device = "/TPU:1"}
               %id2 = "tf.Identity"(%var_handle) : (!tf_res) -> !tf_res
               tf_executor.yield
             }
             tf_executor.fetch %island : !tf_executor.control
          }
          "tf.Yield"() : () -> ()
        }, {
          tf_executor.graph {
             // CHECK: tf_executor.island
             %island = tf_executor.island {
               // CHECK-NEXT: "tf.Identity"
               // CHECK-SAME: {device = "/TPU:0"}
               %id1 = "tf.Identity"(%id0) : (!tf_res) -> !tf_res
               // CHECK-NEXT: "tf.Identity"
               // CHECK-SAME: {device = "/TPU:1"}
               %id2 = "tf.Identity"(%var_handle) : (!tf_res) -> !tf_res
               tf_executor.yield
             }
             tf_executor.fetch %island : !tf_executor.control
          }
          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// Test propagation through WhileRegion (inlined calls)
// CHECK-LABEL: func @propagate_while_region_inlined
func.func @propagate_while_region_inlined(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"} : () -> !tf_res
      // CHECK-NEXT: "tf.WhileRegion"
      "tf.WhileRegion"(%arg1, %id0, %var_handle) ({
          ^bb0(%carg0: tensor<i32>, %carg1: !tf_res, %carg2: !tf_res):
            // CHECK: ^bb
            // CHECK: "tf.Identity"
            // CHECK-SAME: {device = "/TPU:0"}
            %cid0 = "tf.Identity"(%carg1) : (!tf_res) -> !tf_res loc("cid0")
            %read = "tf.ReadVariableOp"(%cid0) : (!tf_res) -> tensor<32xf32>
            %cst = arith.constant dense<3.0> : tensor<32xf32>
            %cmp = "tf.Less"(%read, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xi1>
            %dims = arith.constant dense<0> : tensor<1xi32>
            %reduce = "tf.All"(%cmp, %dims) {keep_dims = false} : (tensor<32xi1>, tensor<1xi32>) -> tensor<i1>
            "tf.Yield"(%reduce) : (tensor<i1>) -> ()
        }, {
          ^bb0(%barg0: tensor<i32>, %barg1: !tf_res, %barg2: !tf_res):
            // CHECK: ^bb
            // CHECK: "tf.Identity"
            // CHECK-SAME: {device = "/TPU:0"}
            %bid0 = "tf.Identity"(%barg1) : (!tf_res) -> !tf_res
            // CHECK-NEXT: "tf.Identity"
            // CHECK-SAME: {device = "/TPU:1"}
            %id1 = "tf.Identity"(%barg2) : (!tf_res) -> !tf_res
            "tf.Yield"(%barg0, %bid0, %id1) : (tensor<i32>, !tf_res,!tf_res) -> ()
        }){is_stateless = false}
        : (tensor<i32>, !tf_res, !tf_res) -> (tensor<i32>, !tf_res, !tf_res)
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// Test propagation through WhileRegion (non-inlined calls)
// CHECK-LABEL: func @propagate_while_region
func.func @propagate_while_region(
  %arg0: !tf_res {tf.device = "/TPU:0"},
  %arg1: tensor<i32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
      // CHECK-NEXT: "tf.VarHandleOp"
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0", device = "/TPU:1"} : () -> !tf_res
      // CHECK-NEXT: "tf.WhileRegion"
      "tf.WhileRegion"(%arg1, %id0, %var_handle) ({
          ^bb0(%carg0: tensor<i32>, %carg1: !tf_res, %carg2: !tf_res):
            %cond = func.call @whileregion_cond(%carg0, %carg1, %carg2) : (tensor<i32>, !tf_res, !tf_res) -> tensor<i1>
            "tf.Yield"(%cond) : (tensor<i1>) -> ()
        }, {
          ^bb0(%barg0: tensor<i32>, %barg1: !tf_res, %barg2: !tf_res):
            %new_values:3 = func.call @whileregion_body(%barg0, %barg1, %barg2) : (tensor<i32>, !tf_res,!tf_res) -> (tensor<i32>, !tf_res,!tf_res)
            "tf.Yield"(%new_values#0, %new_values#1, %new_values#2) : (tensor<i32>, !tf_res,!tf_res) -> ()
        }){is_stateless = false}
        : (tensor<i32>, !tf_res, !tf_res) -> (tensor<i32>, !tf_res, !tf_res)
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @whileregion_body
func.func @whileregion_body(%arg0: tensor<i32>, %arg1: !tf_res, %arg2: !tf_res) -> (tensor<i32>, !tf_res, !tf_res) {
  %graph:3 = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:4 = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg1) : (!tf_res) -> !tf_res
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:1"}
      %id1 = "tf.Identity"(%arg2) : (!tf_res) -> !tf_res
      tf_executor.yield %arg0, %id0, %id1 : tensor<i32>, !tf_res, !tf_res
    }
    tf_executor.fetch %island#0, %island#1, %island#2 : tensor<i32>, !tf_res, !tf_res
  }
  func.return %graph#0, %graph#1, %graph#2: tensor<i32>, !tf_res, !tf_res
}

// CHECK-LABEL: func @whileregion_cond
func.func @whileregion_cond(%arg0: tensor<i32>, %arg1: !tf_res, %arg2: !tf_res) -> tensor<i1> {
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
      // CHECK-NEXT: "tf.Identity"
      // CHECK-SAME: {device = "/TPU:0"}
      %id0 = "tf.Identity"(%arg1) : (!tf_res) -> !tf_res
      %read = "tf.ReadVariableOp"(%id0) : (!tf_res) -> tensor<32xf32>
      %cst = arith.constant dense<3.0> : tensor<32xf32>
      %cmp = "tf.Less"(%read, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xi1>
      %dims = arith.constant dense<0> : tensor<1xi32>
      %reduce = "tf.All"(%cmp, %dims) {keep_dims = false} : (tensor<32xi1>, tensor<1xi32>) -> tensor<i1>
      tf_executor.yield %reduce : tensor<i1>
    }
    tf_executor.fetch %island#0 : tensor<i1>
  }
  func.return %graph : tensor<i1>
}
