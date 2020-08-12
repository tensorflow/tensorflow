// RUN: tf-opt -split-input-file -verify-diagnostics -tf-resource-device-inference %s | FileCheck %s

!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// Tests that the pass can correctly propagate device attributes inside the same
// function.

// CHECK-LABEL: func @propagate_in_function
func @propagate_in_function(
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
  return
}

// -----
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// Tesets that the pass can propagate through tf.If's branches.

// CHECK-LABEL: func @propagate_if_op
func @propagate_if_op(
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
  return
}

// CHECK-LABEL: func @if_then
func @if_then(
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
  return
}

// CHECK-LABEL: func @if_else
func @if_else(
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
  return
}


// -----
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// Tesets that the pass can propagate through tf.While's branches.
// CHECK-LABEL: func @propagate_while_op
func @propagate_while_op(
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
  return
}

// CHECK-LABEL: func @while_body
func @while_body(
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
  return %graph#0, %graph#1, %graph#2
     : tensor<i32>, !tf_res,
       !tf_res
}

// CHECK-LABEL: func @while_cond
func @while_cond(
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
  return %graph : tensor<32xf32>
}

// -----
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// Tesets that the pass reports error on conflicting assignments from multiple
// callers.

func @error_on_conflict_multiple_callers(
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
  return
}

func @if_then_and_else(
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
  return
}

// -----

// Test that the pass can propagate through calls
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @test_function
// CHECK-SAME: {tf.device = "/TPU:0"}
func @test_function(%arg0: !tf_res) {
  // CHECK: "tf.Identity"
  // CHECK-SAME: {device = "/TPU:0"}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  %read = "tf.ReadVariableOp"(%id0) : (!tf_res) -> tensor<32xf32>
  %cst = constant dense<3.0> : tensor<32xf32>
  %add = "tf.AddV2"(%read, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  "tf.AssignVariableOp"(%arg0, %add) : (!tf_res, tensor<32xf32>) -> ()
  return
}

// CHECK-LABEL: func @propagate_through_calls
func @propagate_through_calls(
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
      call @test_function(%id1) : (!tf_res) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// Test propagation through IfRegion (with non-inlined calls)
// CHECK-LABEL: func @propagate_if_region
func @propagate_if_region(
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
          call @ifregion_then(%id0, %var_handle) : (!tf_res, !tf_res) -> ()
          "tf.Yield"() : () -> ()
        }, {
          call @ifregion_else(%id0, %var_handle) : (!tf_res, !tf_res) -> ()
          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @ifregion_then
// CHECK-SAME: (%arg0: {{.+}} {tf.device = "/TPU:0"}, %arg1: {{.+}} {tf.device = "/TPU:1"}
func @ifregion_then(
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
  return
}

// CHECK-LABEL: func @ifregion_else
// CHECK-SAME: (%arg0: {{.+}} {tf.device = "/TPU:0"}, %arg1: {{.+}} {tf.device = "/TPU:1"}
func @ifregion_else(
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
  return
}

// Test progagation through IfRegion (inlined calls)
// CHECK-LABEL: func @propagate_if_region_inlined
func @propagate_if_region_inlined(
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
  return
}
