// RUN: tf-opt -split-input-file -tf-test-resource-alias-analysis -verify-diagnostics %s | FileCheck %s

// Test 2 resources that do not alias.

!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-LABEL: func @non_aliasing_reads_writes
// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func @non_aliasing_reads_writes(
  %arg0: !tf_res,
  %arg1: !tf_res,
  %arg2: tensor<32xf32>) -> (tensor<32xf32>) {
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
      %read0 = "tf.ReadVariableOp"(%arg0) : (!tf_res) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg0, %arg2) : (!tf_res, tensor<32xf32>) -> ()
      %read1 = "tf.ReadVariableOp"(%arg1) : (!tf_res) -> tensor<32xf32>
      // expected-remark@below {{Result #0, ID 0 : 0}}
      %var_handle = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
      %read2 = "tf.ReadVariableOp"(%var_handle) : (!tf_res) -> tensor<32xf32>
      "tf.AssignVariableOp"(%arg1, %read0) : (!tf_res, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg0, %read2) : (!tf_res, tensor<32xf32>) -> ()
      %read3 = "tf.ReadVariableOp"(%arg0) : (!tf_res) -> tensor<32xf32>
      tf_executor.yield %read3 : tensor<32xf32>
    }
    tf_executor.fetch %island#0 : tensor<32xf32>
  }
  return %graph : tensor<32xf32>
}

// -----
// Tests aliasing of the two resource handles that refer to the same variable.

!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-LABEL: func @aliasing_reads_writes
func @aliasing_reads_writes(%arg0: tensor<32xf32>) -> () {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // expected-remark@below {{Result #0, ID 0 : 0, 1, 2}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
      // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
      // expected-remark@below {{Result #0, ID 2 : 0, 1, 2}}
      %vh1_id:2 = "tf.IdentityN"(%vh1, %arg0) : (!tf_res, tensor<32xf32>) -> (!tf_res, tensor<32xf32>)
      %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<32xf32>
      "tf.AssignVariableOp"(%vh1_id#0, %arg0) : (!tf_res, tensor<32xf32>) -> ()
      %read1 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<32xf32>
      %read2 = "tf.ReadVariableOp"(%vh1) : (!tf_res) -> tensor<32xf32>
      "tf.AssignVariableOp"(%vh0, %read2) : (!tf_res, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%vh1_id#0, %read1) : (!tf_res, tensor<32xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// -----
// Test an unknown op that has a resource result is marked unknown

!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-LABEL: func @unknown_resource_op
func @unknown_resource_op(%arg0: tensor<32xf32>) -> () {
    // expected-remark@below {{Result #0, ID 0 : Unknown}}
    %0 = "tf.UnknownVarHandleOp"() : () -> !tf_res
}

// -----
// Test aliasing through IfOp

!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @if_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
func @if_op_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<32xf32>
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 1, 2, 3, 5}}
  // expected-remark@below {{Result #2, ID 3 : 0, 1, 2, 3, 5}}
  %if:3 = "tf.If"(%read0, %arg1, %vh0) {
            then_branch = @if_then, else_branch = @if_else, is_stateless = true
          } : (tensor<32xf32>, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  return
}

// expected-remark@below {{Region #0, Arg #0, ID 2 : 0, 1, 2}}
// expected-remark@below {{Region #0, Arg #1, ID 3 : 0, 3}}
func @if_then(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  return %u0, %id0, %id0 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func @if_else(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  return %id0, %id0, %arg1 : !tf_res, !tf_res, !tf_res
}

// -----
// Test aliasing through CaseOp

!tf_res = type tensor<*x!tf.resource<tensor<i32>>>

// CHECK-LABEL: func @case_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
func @case_op_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<i32>
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 1, 2, 3, 5}}
  // expected-remark@below {{Result #2, ID 3 : 0, 1, 2, 3, 5}}
  %if:3 = "tf.Case"(%read0, %arg1, %vh0) {
            branches = [@case_branch0, @case_branch1, @case_branch2],
            is_stateless = true
          } : (tensor<i32>, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  return
}

// expected-remark@below {{Region #0, Arg #0, ID 2 : 0, 1, 2}}
// expected-remark@below {{Region #0, Arg #1, ID 3 : 0, 3}}
func @case_branch0(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  return %u0, %id0, %id0 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func @case_branch1(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  return %id0, %id0, %arg1 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 0 : 0}}
// expected-remark@below {{Region #0, Arg #1, ID 1 : 1}}
func @case_branch2(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  return %arg0, %arg0, %arg1 : !tf_res, !tf_res, !tf_res
}

// -----
// Test aliasing through WhileOp
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @while_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
// expected-remark@below {{Region #0, Arg #2, ID 6 : 1, 2, 3, 6}}
func @while_op_aliasing(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 1, 2, 3, 5, 6}}
  // expected-remark@below {{Result #2, ID 3 : 1, 2, 3, 5, 6}}
  %w:3 = "tf.While"(%arg0, %arg1, %arg2) {
            body = @while_body, cond = @while_cond, is_stateless = false
         } : (!tf_res, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  return
}

// CHECK-LABEL: func @while_body
// Return 0 : new unknown resource
// Return 1 : arg2
// Return 2 : arg1
// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 0, 2}}
// expected-remark@below {{Region #0, Arg #2, ID 3 : 0, 3}}
func @while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  return %u0, %arg2, %arg1 : !tf_res, !tf_res, !tf_res
}

// CHECK-LABEL: func @while_cond
// expected-remark@below {{Region #0, Arg #0, ID 0 : 0}}
// expected-remark@below {{Region #0, Arg #1, ID 1 : 1}}
// expected-remark@below {{Region #0, Arg #2, ID 2 : 2}}
func @while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) -> tensor<i1> {
  %0 = constant dense<false> : tensor<i1>
  return %0 : tensor<i1>
}

// -----
// Test alias propagation through calls.
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-LABEL: func @aliasing_through_calls
func @aliasing_through_calls(%arg0: tensor<32xf32>) -> () {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 2, 3}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2, 3}}
  %vh1 = "tf.Identity"(%vh0) : (!tf_res) -> (!tf_res)
  // expected-remark@below {{Result #0, ID 2 : Unknown}}
  // expected-remark@below {{Result #1, ID 3 : 0, 1, 2, 3}}
  %c:2 = call @passthru(%vh1) : (!tf_res) -> (!tf_res, !tf_res)
  return
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
func @passthru(%arg0: !tf_res) -> (!tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vx = "tf.VarHandleOp"() {container = "cf", shared_name = "vx"} : () -> !tf_res
  return %vx, %arg0 : !tf_res, !tf_res
}

// -----
// Test aliasing through IfRegion

!tf_res = type tensor<*x!tf.resource<tensor<i1>>>

// CHECK-LABEL: func @if_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 7 : 1, 4, 6, 7}}
// expected-remark@below {{Region #0, Arg #1, ID 8 : 1, 2, 4, 5, 6, 8}}
func @if_region_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 3, 4, 5}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<i1>
  // expected-remark@below {{Result #0, ID 4 : Unknown}}
  // expected-remark@below {{Result #1, ID 5 : 0, 1, 2, 3, 4, 5, 6, 8}}
  // expected-remark@below {{Result #2, ID 6 : 1, 2, 4, 5, 6, 7, 8}}
  %if:3 = "tf.IfRegion"(%read0) ({
            // expected-remark@below {{Result #0, ID 1 : Unknown}}
            %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
            // expected-remark@below {{Result #0, ID 2 : 1, 2, 4, 5, 6, 8}}
            %id0 = "tf.Identity"(%arg1) : (!tf_res) -> !tf_res
            "tf.Yield"(%u0, %id0, %id0) : (!tf_res, !tf_res, !tf_res) -> ()
          }, {
            // expected-remark@below {{Result #0, ID 3 : 0, 1, 3, 4, 5}}
            %id0 = "tf.Identity"(%vh0) : (!tf_res) -> !tf_res
            "tf.Yield"(%id0, %id0, %arg0) : (!tf_res, !tf_res, !tf_res) -> ()
          }) {is_stateless = true} : (tensor<i1>) -> (!tf_res, !tf_res, !tf_res)
  return
}

// -----
// Test aliasing through CaseRegion

!tf_res = type tensor<*x!tf.resource<tensor<i32>>>

// CHECK-LABEL: func @case_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 7 : 1, 4, 6, 7}}
// expected-remark@below {{Region #0, Arg #1, ID 8 : 1, 2, 4, 5, 6, 8}}
func @case_region_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 3, 4, 5}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<i32>
  // expected-remark@below {{Result #0, ID 4 : Unknown}}
  // expected-remark@below {{Result #1, ID 5 : 0, 1, 2, 3, 4, 5, 6, 8}}
  // expected-remark@below {{Result #2, ID 6 : 1, 2, 4, 5, 6, 7, 8}}
  %if:3 = "tf.CaseRegion"(%read0) ({
            // expected-remark@below {{Result #0, ID 1 : Unknown}}
            %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
            // expected-remark@below {{Result #0, ID 2 : 1, 2, 4, 5, 6, 8}}
            %id0 = "tf.Identity"(%arg1) : (!tf_res) -> !tf_res
            "tf.Yield"(%u0, %id0, %id0) : (!tf_res, !tf_res, !tf_res) -> ()
          }, {
            // expected-remark@below {{Result #0, ID 3 : 0, 1, 3, 4, 5}}
            %id0 = "tf.Identity"(%vh0) : (!tf_res) -> !tf_res
            "tf.Yield"(%id0, %id0, %arg0) : (!tf_res, !tf_res, !tf_res) -> ()
          }, {
            "tf.Yield"(%vh0, %arg1, %arg1) : (!tf_res, !tf_res, !tf_res) -> ()
          }) {is_stateless = true} : (tensor<i32>) -> (!tf_res, !tf_res, !tf_res)
  return
}

// -----
// Test aliasing through WhileRegion
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @while_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 11 : 1, 8, 11}}
// expected-remark@below {{Region #0, Arg #1, ID 12 : 1, 8, 9, 10, 12}}
// expected-remark@below {{Region #0, Arg #2, ID 13 : 1, 8, 9, 10, 13}}
func @while_region_aliasing(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 8}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 8 : Unknown}}
  // expected-remark@below {{Result #1, ID 9 : 1, 8, 9, 10, 12, 13}}
  // expected-remark@below {{Result #2, ID 10 : 1, 8, 9, 10, 12, 13}}
  // expected-remark@below {{Region #0, Arg #0, ID 2 : 1, 2, 8}}
  // expected-remark@below {{Region #0, Arg #1, ID 3 : 1, 3, 8}}
  // expected-remark@below {{Region #0, Arg #2, ID 4 : 1, 4, 8}}
  // expected-remark@below {{Region #1, Arg #0, ID 5 : 1, 5, 8}}
  // expected-remark@below {{Region #1, Arg #1, ID 6 : 1, 6, 8}}
  // expected-remark@below {{Region #1, Arg #2, ID 7 : 1, 7, 8}}
  %w:3 = "tf.WhileRegion"(%arg0, %arg1, %arg2) ({
          ^bb0(%carg0: !tf_res, %carg1: !tf_res, %carg2: !tf_res):
          %0 = constant dense<false> : tensor<i1>
          "tf.Yield"(%0) : (tensor<i1>) -> ()
         },{
          ^bb0(%barg0: !tf_res, %barg1: !tf_res, %barg2: !tf_res):
          // expected-remark@below {{Result #0, ID 1 : Unknown}}
          %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
          "tf.Yield"(%u0, %barg2, %barg1) : (!tf_res, !tf_res, !tf_res) -> ()
         }) {is_stateless = false} : (!tf_res, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  return
}

// -----
// Test aliasing through calls
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_calls
func @aliasing_through_calls(%arg0: tensor<32xf32>) -> () {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 2}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 0, 1, 2}}
  %c:2 = call @passthru(%vh0) : (!tf_res) -> (!tf_res, !tf_res)
  return
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
func @passthru(%arg0: !tf_res) -> (!tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  return %vh0, %arg0 : !tf_res, !tf_res
}

// -----
// Test aliasing through tf_device.launch
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_launch
func @aliasing_through_launch(%arg0: tensor<32xf32>) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %vh = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> !tf_res

  // expected-remark@below {{Result #0, ID 1 : 0, 1}}
  %launch = "tf_device.launch"() ({
    tf_device.return %vh : !tf_res
  }) {device = ""} : () -> !tf_res
  return
}

// -----
// Test aliasing through tf_device.cluster
!tf_res = type tensor<*x!tf.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_cluster
func @aliasing_through_cluster(%arg0: tensor<32xf32>) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %vh = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> !tf_res

  // expected-remark@below {{Result #0, ID 1 : 0, 1}}
  %cluster = "tf_device.cluster"() ({
    tf_device.return %vh : !tf_res
  }) : () -> !tf_res
  return
}
