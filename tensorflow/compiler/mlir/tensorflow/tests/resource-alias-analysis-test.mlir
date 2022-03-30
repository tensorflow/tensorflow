// RUN: tf-opt -split-input-file -tf-test-resource-alias-analysis -verify-diagnostics %s | FileCheck %s

// Test 2 resources that do not alias.

!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-LABEL: func @non_aliasing_reads_writes
// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func.func @non_aliasing_reads_writes(
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
  func.return %graph : tensor<32xf32>
}

// -----
// Tests aliasing of the two resource handles that refer to the same variable.

!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-LABEL: func @aliasing_reads_writes
func.func @aliasing_reads_writes(%arg0: tensor<32xf32>) -> () {
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
  func.return
}

// -----
// Test an unknown op that has a resource result is marked unknown

!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-LABEL: func @unknown_resource_op
func.func @unknown_resource_op(%arg0: tensor<32xf32>) -> () {
    // expected-remark@below {{Result #0, ID 0 : Unknown}}
    %0 = "tf.UnknownVarHandleOp"() : () -> !tf_res
}

// -----
// Test aliasing through TPUReplicatedInput
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-LABEL: func @aliasing_tpu_replicated_input
func.func @aliasing_tpu_replicated_input(%arg0: tensor<32xf32>) -> () {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // expected-remark@below {{Result #0, ID 0 : 0, 2}}
      %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
      // expected-remark@below {{Result #0, ID 1 : 1, 2}}
      %vh1 = "tf.VarHandleOp"() {container = "c", shared_name = "v1"} : () -> !tf_res
      // expected-remark@below {{Result #0, ID 2 : 0, 1, 2}}
      %replicated = "tf.TPUReplicatedInput"(%vh0, %vh1) : (!tf_res, !tf_res) -> (!tf_res)
      "tf.AssignVariableOp"(%vh0, %arg0) : (!tf_res, tensor<32xf32>) -> ()
      %read1 = "tf.ReadVariableOp"(%replicated) : (!tf_res) -> tensor<32xf32>
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  func.return
}

// -----
// Test aliasing through IfOp

!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @if_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
func.func @if_op_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  %read0 = "tf.ReadVariableOp"(%vh0) : (!tf_res) -> tensor<32xf32>
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 1, 2, 3, 5}}
  // expected-remark@below {{Result #2, ID 3 : 0, 1, 2, 3, 5}}
  %if:3 = "tf.If"(%read0, %arg1, %vh0) {
            then_branch = @if_then, else_branch = @if_else, is_stateless = true
          } : (tensor<32xf32>, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  func.return
}

// expected-remark@below {{Region #0, Arg #0, ID 2 : 0, 1, 2}}
// expected-remark@below {{Region #0, Arg #1, ID 3 : 0, 3}}
func.func @if_then(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  func.return %u0, %id0, %id0 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func.func @if_else(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  func.return %id0, %id0, %arg1 : !tf_res, !tf_res, !tf_res
}

// -----
// Test aliasing through CaseOp

!tf_res = type tensor<*x!tf_type.resource<tensor<i32>>>

// CHECK-LABEL: func @case_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
func.func @case_op_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
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
  func.return
}

// expected-remark@below {{Region #0, Arg #0, ID 2 : 0, 1, 2}}
// expected-remark@below {{Region #0, Arg #1, ID 3 : 0, 3}}
func.func @case_branch0(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  func.return %u0, %id0, %id0 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 2}}
func.func @case_branch1(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %id0 = "tf.Identity"(%arg0) : (!tf_res) -> !tf_res
  func.return %id0, %id0, %arg1 : !tf_res, !tf_res, !tf_res
}

// expected-remark@below {{Region #0, Arg #0, ID 0 : 0}}
// expected-remark@below {{Region #0, Arg #1, ID 1 : 1}}
func.func @case_branch2(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  func.return %arg0, %arg0, %arg1 : !tf_res, !tf_res, !tf_res
}

// -----
// Test aliasing through WhileOp
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @while_op_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 4 : 1, 4}}
// expected-remark@below {{Region #0, Arg #1, ID 5 : 1, 2, 3, 5}}
// expected-remark@below {{Region #0, Arg #2, ID 6 : 1, 2, 3, 6}}
func.func @while_op_aliasing(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 1, 2, 3, 5, 6}}
  // expected-remark@below {{Result #2, ID 3 : 1, 2, 3, 5, 6}}
  %w:3 = "tf.While"(%arg0, %arg1, %arg2) {
            body = @while_body, cond = @while_cond, is_stateless = false
         } : (!tf_res, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  func.return
}

// CHECK-LABEL: func @while_body
// Return 0 : new unknown resource
// Return 1 : arg2
// Return 2 : arg1
// expected-remark@below {{Region #0, Arg #0, ID 1 : 0, 1}}
// expected-remark@below {{Region #0, Arg #1, ID 2 : 0, 2}}
// expected-remark@below {{Region #0, Arg #2, ID 3 : 0, 3}}
func.func @while_body(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) -> (!tf_res, !tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : Unknown}}
  %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
  func.return %u0, %arg2, %arg1 : !tf_res, !tf_res, !tf_res
}

// CHECK-LABEL: func @while_cond
// expected-remark@below {{Region #0, Arg #0, ID 0 : 0}}
// expected-remark@below {{Region #0, Arg #1, ID 1 : 1}}
// expected-remark@below {{Region #0, Arg #2, ID 2 : 2}}
func.func @while_cond(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) -> tensor<i1> {
  %0 = arith.constant dense<false> : tensor<i1>
  func.return %0 : tensor<i1>
}

// -----
// Test alias propagation through calls.
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-LABEL: func @aliasing_through_calls
func.func @aliasing_through_calls(%arg0: tensor<32xf32>) -> () {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 2, 3}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1, 2, 3}}
  %vh1 = "tf.Identity"(%vh0) : (!tf_res) -> (!tf_res)
  // expected-remark@below {{Result #0, ID 2 : Unknown}}
  // expected-remark@below {{Result #1, ID 3 : 0, 1, 2, 3}}
  %c:2 = func.call @passthru(%vh1) : (!tf_res) -> (!tf_res, !tf_res)
  func.return
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
func.func @passthru(%arg0: !tf_res) -> (!tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vx = "tf.VarHandleOp"() {container = "cf", shared_name = "vx"} : () -> !tf_res
  func.return %vx, %arg0 : !tf_res, !tf_res
}

// -----
// Test aliasing through IfRegion

!tf_res = type tensor<*x!tf_type.resource<tensor<i1>>>

// CHECK-LABEL: func @if_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 7 : 1, 4, 6, 7}}
// expected-remark@below {{Region #0, Arg #1, ID 8 : 1, 2, 4, 5, 6, 8}}
func.func @if_region_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
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
  func.return
}

// -----
// Test aliasing through CaseRegion

!tf_res = type tensor<*x!tf_type.resource<tensor<i32>>>

// CHECK-LABEL: func @case_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 7 : 1, 4, 6, 7}}
// expected-remark@below {{Region #0, Arg #1, ID 8 : 1, 2, 4, 5, 6, 8}}
func.func @case_region_aliasing(%arg0: !tf_res, %arg1: !tf_res) {
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
  func.return
}

// -----
// Test aliasing through WhileRegion
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @while_region_aliasing
// expected-remark@below {{Region #0, Arg #0, ID 11 : 1, 8, 11}}
// expected-remark@below {{Region #0, Arg #1, ID 12 : 1, 8, 9, 10, 12}}
// expected-remark@below {{Region #0, Arg #2, ID 13 : 1, 8, 9, 10, 13}}
func.func @while_region_aliasing(%arg0: !tf_res, %arg1: !tf_res, %arg2: !tf_res) {
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
          %0 = arith.constant dense<false> : tensor<i1>
          "tf.Yield"(%0) : (tensor<i1>) -> ()
         },{
          ^bb0(%barg0: !tf_res, %barg1: !tf_res, %barg2: !tf_res):
          // expected-remark@below {{Result #0, ID 1 : Unknown}}
          %u0 = "tf._UnknownSideEffectingOp_"() : () -> !tf_res
          "tf.Yield"(%u0, %barg2, %barg1) : (!tf_res, !tf_res, !tf_res) -> ()
         }) {is_stateless = false} : (!tf_res, !tf_res, !tf_res) -> (!tf_res, !tf_res, !tf_res)
  func.return
}

// -----
// Test aliasing through calls
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_calls
func.func @aliasing_through_calls(%arg0: tensor<32xf32>) -> () {
  // expected-remark@below {{Result #0, ID 0 : 0, 1, 2}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : Unknown}}
  // expected-remark@below {{Result #1, ID 2 : 0, 1, 2}}
  %c:2 = func.call @passthru(%vh0) : (!tf_res) -> (!tf_res, !tf_res)
  func.return
}

// expected-remark@below {{Region #0, Arg #0, ID 1 : 1}}
func.func @passthru(%arg0: !tf_res) -> (!tf_res, !tf_res) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %vh0 = "tf.VarHandleOp"() {container = "c", shared_name = "v0"} : () -> !tf_res
  func.return %vh0, %arg0 : !tf_res, !tf_res
}

// -----
// Test aliasing through tf_device.launch
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_launch
func.func @aliasing_through_launch(%arg0: tensor<32xf32>) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %vh = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> !tf_res

  // expected-remark@below {{Result #0, ID 1 : 0, 1}}
  %launch = "tf_device.launch"() ({
    tf_device.return %vh : !tf_res
  }) {device = ""} : () -> !tf_res
  func.return
}

// -----
// Test aliasing through tf_device.cluster
!tf_res = type tensor<*x!tf_type.resource<tensor<32xf32>>>

// CHECK-LABEL: func @aliasing_through_cluster
func.func @aliasing_through_cluster(%arg0: tensor<32xf32>) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %vh = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> !tf_res

  // expected-remark@below {{Result #0, ID 1 : 0, 1}}
  %cluster = "tf_device.cluster"() ({
    tf_device.return %vh : !tf_res
  }) : () -> !tf_res
  func.return
}

// -----

// Tests that ops with trait `TF_UniqueResourceAllocation` are not aliasing.

// CHECK-LABEL: func @unique_resource_allocation
func.func @unique_resource_allocation(%arg0: tensor<i32>) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %stack_handle1 = "tf.StackV2"(%arg0) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf_type.resource>
  // expected-remark@below {{Result #0, ID 1 : 1}}
  %stack_handle2 = "tf.StackV2"(%arg0) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf_type.resource>
  func.return
}

// -----

!tf_res = type tensor<*x!tf_type.resource<tensor<f32>>>

// Tests that ops with different known resource types get different resource IDs
// assigned, even if resource instances are unknown.
func.func @known_different_resource_types_unknown_instances(%arg0: tensor<i32>) {
  // expected-remark@below {{Result #0, ID 0 : 0}}
  %iter_handle = "tf.IteratorV2"() {container = "c", shared_name = "v0", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 1}}
  %seed_handle = "tf.DummySeedGenerator"() : () -> !tf_res
  func.return
}

// -----

!tf_res = type tensor<*x!tf_type.resource<tensor<f32>>>

// Tests that ops with same known resource type get same resource ID assigned
// (not unknown ID) if resource instances are unknown.
func.func @known_same_resource_types_unknown_instances(%arg0: tensor<i32>) {
  // expected-remark@below {{Result #0, ID 0 : 0, 1}}
  %iter_handle1 = "tf.IteratorV2"() {container = "c", shared_name = "v0", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
  // expected-remark@below {{Result #0, ID 1 : 0, 1}}
  %iter_handle2 = "tf.IteratorV2"() {container = "c", shared_name = "v1", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
  func.return
}

// -----

!tf_res = type tensor<*x!tf_type.resource<tensor<f32>>>

// Tests that an allocated resource is correctly propagated to island and graph
// results.
func.func @allocated_resource_propagation_island_graph() {
  // expected-remark@below {{Result #0, ID 2 : 0, 1, 2}}
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    // expected-remark@below {{Result #0, ID 1 : 0, 1, 2}}
    %island:2 = tf_executor.island {
      // expected-remark@below {{Result #0, ID 0 : 0, 1, 2}}
      %iter_handle = "tf.IteratorV2"() {container = "c", shared_name = "v0", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
      tf_executor.yield %iter_handle : !tf_res
    }
    tf_executor.fetch %island#0 : !tf_res
  }
  func.return
}

// -----

!tf_res = type tensor<*x!tf_type.resource<tensor<f32>>>

// Tests that aliasing and non-aliasing values are correctly identified through
// multiple islands (`%iter_handle1`, `%iter_handle2`, `%island1#0` and
// `%island3#0` all point to the same resource here).
func.func @multiple_islands() {
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    // expected-remark@below {{Result #0, ID 2 : 0, 2, 3, 4}}
    %island1:2 = tf_executor.island {
      // expected-remark@below {{Result #0, ID 0 : 0, 2, 3, 4}}
      %iter_handle1 = "tf.IteratorV2"() {container = "c", shared_name = "v0", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
      // expected-remark@below {{Result #0, ID 1 : 1}}
      %seed_handle = "tf.DummySeedGenerator"() : () -> !tf_res
      tf_executor.yield %iter_handle1 : !tf_res
    }
    %island2:2 = tf_executor.island {
      %1 = "tf.IteratorGetNext"(%island1#0) : (!tf_res) -> tensor<f32>
      tf_executor.yield %1 : tensor<f32>
    }
    // expected-remark@below {{Result #0, ID 4 : 0, 2, 3, 4}}
    %island3:2 = tf_executor.island {
      // expected-remark@below {{Result #0, ID 3 : 0, 2, 3, 4}}
      %iter_handle2 = "tf.IteratorV2"() {container = "c", shared_name = "v0", output_shapes = [#tf_type.shape<>], output_types = [!tf_res]} : () -> !tf_res
      tf_executor.yield %iter_handle2 : !tf_res
    }
    tf_executor.fetch %island2#0 : tensor<f32>
  }
  func.return
}
