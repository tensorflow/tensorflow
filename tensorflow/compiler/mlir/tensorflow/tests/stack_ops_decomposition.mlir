// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-stack-ops-decomposition | FileCheck %s -dump-input-on-failure

// Tests simple scalar stack operations without control flow.

// CHECK-LABEL: func @main
func @main() -> tensor<f32> {
  // CHECK-NEXT: "tf.Const"() {value = dense<10> : tensor<i32>}
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[ZERO_SCALAR:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[CAST_ZERO:.*]] = "tf.Cast"(%[[ZERO_SCALAR]]) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: %[[CONST10:.*]] = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[CAST_ZERO]], %[[CONST10]]) : (tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[BUFFER:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<10xf32>>>
  // CHECK-NEXT: %[[SIZE:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<1xi32>>>
  // CHECK-NEXT: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[SIZE]], %[[ZERO]])
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[BUFFER]], %[[BROADCAST]])
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %id = "tf.Identity"(%stack) : (tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK-NEXT: %[[PUSHVAL:.*]] = "tf._SomeOp"()
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NEXT: %[[READ_VAL:.*]] = "tf.ReadVariableOp"(%[[BUFFER]])
  // CHECK-NEXT: %[[READ_SIZE:.*]] = "tf.ReadVariableOp"(%[[SIZE]])
  // CHECK-NEXT: %[[UPDATE_SHAPE:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[UPDATE_SLICE:.*]] = "tf.Reshape"(%[[PUSHVAL]], %[[UPDATE_SHAPE]]) : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ_VAL]], %[[UPDATE_SLICE]], %[[READ_SIZE]]) : (tensor<10xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<10xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[BUFFER]], %[[UPDATE]]) : (tensor<!tf.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
  // CHECK-NEXT: %[[CONST1:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[NEW_SIZE:.*]] = "tf.AddV2"(%[[READ_SIZE]], %[[CONST1]]) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[SIZE]], %[[NEW_SIZE]]) : (tensor<!tf.resource<tensor<1xi32>>>, tensor<1xi32>) -> ()
  %push = "tf.StackPushV2"(%id, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  %pop = "tf.StackPopV2"(%stack) : (tensor<!tf.resource>) -> tensor<f32>
  // CHECK-NEXT: %[[READ_VAL1:.*]] = "tf.ReadVariableOp"(%[[BUFFER]])
  // CHECK-NEXT: %[[READ_SIZE1:.*]] = "tf.ReadVariableOp"(%[[SIZE]])
  // CHECK-NEXT: %[[CONST1_1:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SUB:.*]] = "tf.Sub"(%[[READ_SIZE1]], %[[CONST1_1]])
  // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = "tf.Slice"(%[[READ_VAL1]], %[[SUB]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[ELEM_SHAPE:.*]] = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NEXT: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[SIZE]], %[[SUB]]) : (tensor<!tf.resource<tensor<1xi32>>>, tensor<1xi32>) -> ()
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK-NEXT:  return %[[ELEM]] : tensor<f32>
  return %pop : tensor<f32>
}

// -----

// Tests simple non-scalar stack operations without control flow.

// CHECK-LABEL: func @main
func @main() -> tensor<2xi32> {
  // CHECK-NEXT: "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[STACK_SHAPE:.*]] = "tf.Const"() {value = dense<[10, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[ZERO_CONST]], %[[STACK_SHAPE]]) : (tensor<i32>, tensor<2xi32>) -> tensor<10x2xi32>
  // CHECK-NEXT: %[[BUFFER:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<10x2xi32>>>
  // CHECK-NEXT: %[[SIZE:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<1xi32>>>
  // CHECK-NEXT: %[[ZERO_SIZE:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[SIZE]], %[[ZERO_SIZE]]) : (tensor<!tf.resource<tensor<1xi32>>>, tensor<1xi32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[BUFFER]], %[[BROADCAST]]) : (tensor<!tf.resource<tensor<10x2xi32>>>, tensor<10x2xi32>) -> ()
  %stack = "tf.StackV2"(%size) {elem_type = i32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  // CHECK-NEXT: %[[PUSH_VAL:.*]] = "tf._SomeOp"() : () -> tensor<2xi32>
  %elem = "tf._SomeOp"() : () -> tensor<2xi32>
  // CHECK-NEXT: %[[STACK_VAL:.*]] = "tf.ReadVariableOp"(%[[BUFFER]]) : (tensor<!tf.resource<tensor<10x2xi32>>>) -> tensor<10x2xi32>
  // CHECK-NEXT: %[[STACK_SIZE:.*]] = "tf.ReadVariableOp"(%[[SIZE]]) : (tensor<!tf.resource<tensor<1xi32>>>) -> tensor<1xi32>
  // CHECK-NEXT: %[[UPDATE_SHAPE:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK-NEXT: %[[UPDATE_SLICE:.*]] = "tf.Reshape"(%[[PUSH_VAL]], %[[UPDATE_SHAPE]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: %[[ZERO_INDS:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[CONCAT_DIM:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[CONCAT_OFFETS:.*]] = "tf.ConcatV2"(%[[STACK_SIZE]], %[[ZERO_INDS]], %[[CONCAT_DIM]]) : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"(%[[STACK_VAL]], %[[UPDATE_SLICE]], %[[CONCAT_OFFETS]]) : (tensor<10x2xi32>, tensor<1x2xi32>, tensor<2xi32>) -> tensor<10x2xi32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[BUFFER]], %[[UPDATE]]) : (tensor<!tf.resource<tensor<10x2xi32>>>, tensor<10x2xi32>) -> ()
  // CHECK-NEXT: %[[CONST1:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %[[NEW_SIZE:.*]] = "tf.AddV2"(%[[STACK_SIZE]], %[[CONST1]]) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[SIZE]], %[[NEW_SIZE]]) : (tensor<!tf.resource<tensor<1xi32>>>, tensor<1xi32>) -> ()
  %push = "tf.StackPushV2"(%stack, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<2xi32>) -> tensor<2xi32>
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK-NEXT: return %[[PUSH_VAL]] : tensor<2xi32>
  return %push : tensor<2xi32>
}

// -----

// Tests while loop.

// CHECK-LABEL: func @main
func @main() -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.Stack
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %1:2 = "tf.While"(%stack, %max_size) {
    body = @while_body, cond = @while_cond, device = "", is_stateless = false}
       : (tensor<!tf.resource>, tensor<i32>) -> (tensor<!tf.resource>, tensor<i32>)
  // CHECK: "tf.Slice"
  %pop = "tf.StackPopV2"(%1#0) : (tensor<!tf.resource>) -> tensor<f32>
  // CHECK-NOT: tf.Stack
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK: return
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[BARG1:.*]]: tensor<i32>, %[[BARG2:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
func @while_body(%arg0: tensor<!tf.resource>, %arg1: tensor<i32>) -> (tensor<!tf.resource>, tensor<i32>) {
  // CHECK: %[[CONST1:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[BARG1]], %[[CONST1]])
  %sub = "tf.Sub"(%arg1, %const1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NOT: "tf.StackPushV2"
  // CHECK: "tf.XlaDynamicUpdateSlice"
  // CHECK-NOT: "tf.StackPushV2"
  %push = "tf.StackPushV2"(%arg0, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[BARG0]], %[[SUB]], %[[BARG2]]
  return %arg0, %sub : tensor<!tf.resource>, tensor<i32>
}
// CHECK: func @while_cond(%[[CARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[CARG1:.*]]: tensor<i32>, %[[CARG2:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
func @while_cond(%arg0: tensor<!tf.resource>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return %[[CARG1]]
  return %arg1 : tensor<i32>
}

// -----

// Tests IfOp.

// CHECK-LABEL: func @main
func @main(%arg0: tensor<i1>) -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.Stack
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %if_op = "tf.If"(%arg0, %stack) {then_branch = @if_then, else_branch = @if_else, is_stateless = false}
    : (tensor<i1>, tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: "tf.Slice"
  %pop = "tf.StackPopV2"(%if_op) : (tensor<!tf.resource>) -> tensor<f32>
  // CHECK-NOT: tf.Stack
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK: return
  return
}
// CHECK: func @if_then(%[[TARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[TARG1:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
func @if_then(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> {
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NOT: "tf.StackPushV2"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: "tf.AssignVariableOp"(%[[TARG0:.*]], %[[UPDATE]])
  // CHECK: "tf.AssignVariableOp"(%[[EARG1:.*]],
  // CHECK-NOT: "tf.StackPushV2"
  %push = "tf.StackPushV2"(%arg0, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}
// CHECK: func @if_else(%[[EARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[EARG1:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
func @if_else(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> {
  // CHECK-NOT: "tf.StackPopV2"
  // CHECK: "tf.Slice"
  // CHECK: "tf.AssignVariableOp"(%[[EARG1:.*]],
  // CHECK-NOT: "tf.StackPopV2"
  %pop = "tf.StackPopV2"(%arg0) : (tensor<!tf.resource>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}

// -----

// Tests PartitionedCall/StatefulPartitionedCall.

// CHECK-LABEL: func @main
func @main(%arg0: tensor<i1>) -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.Stack
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  // CHECK: "tf.StatefulPartitionedCall"
  // CHECK-SAME: f = @callee_stack_decomposed
  %call = "tf.StatefulPartitionedCall"(%stack, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>, tensor<i1>) -> tensor<!tf.resource>
  // CHECK: "tf.PartitionedCall"
  // CHECK-SAME: f = @callee_stack_decomposed
  %call2 = "tf.PartitionedCall"(%stack, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>, tensor<i1>) -> tensor<!tf.resource>
  // CHECK: "tf.Slice"
  %pop = "tf.StackPopV2"(%call) : (tensor<!tf.resource>) -> tensor<f32>
  // CHECK-NOT: tf.Stack
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK: return
  return
}

// CHECK: func @callee(%[[AARG0:.*]]: tensor<!tf.resource>, %[[AARG1:.*]]: tensor<i1>) -> tensor<!tf.resource>
func @callee(%arg0: tensor<!tf.resource>, %arg1: tensor<i1>) -> tensor<!tf.resource> attributes {sym_visibility = "public"} {
  %elem = "tf._SomeOp"(%arg1) : (tensor<i1>) -> tensor<f32>
  // CHECK: tf.StackPushV2"
  %push = "tf.StackPushV2"(%arg0, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}

// CHECK: func @callee_stack_decomposed(%[[ARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[ARG1:.*]]: tensor<i1>, %[[ARG2:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
// CHECK-NOT: "tf.StackPushV2"
// CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
// CHECK: "tf.AssignVariableOp"(%[[TARG0:.*]], %[[UPDATE]])
// CHECK: "tf.AssignVariableOp"(%[[EARG1:.*]],
// CHECK-NOT: "tf.StackPushV2"

// -----

// Tests PartitionedCall/StatefulPartitionedCall with private callee function.

// CHECK-LABEL: func @main
func @main(%arg0: tensor<i1>) -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.Stack
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  // CHECK: "tf.StatefulPartitionedCall"
  // CHECK-SAME: f = @callee
  %call = "tf.StatefulPartitionedCall"(%stack, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>, tensor<i1>) -> tensor<!tf.resource>
  // CHECK: "tf.PartitionedCall"
  // CHECK-SAME: f = @callee
  %call2 = "tf.PartitionedCall"(%stack, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>, tensor<i1>) -> tensor<!tf.resource>
  // CHECK: "tf.Slice"
  %pop = "tf.StackPopV2"(%call) : (tensor<!tf.resource>) -> tensor<f32>
  // CHECK-NOT: tf.Stack
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK: return
  return
}

// CHECK: func @callee(%[[ARG0:.*]]: tensor<!tf.resource<tensor<10xf32>>>, %[[ARG1:.*]]: tensor<i1>, %[[ARG2:.*]]: tensor<!tf.resource<tensor<1xi32>>>)
func @callee(%arg0: tensor<!tf.resource>, %arg1: tensor<i1>) -> tensor<!tf.resource> attributes {sym_visibility = "private"} {
  %elem = "tf._SomeOp"(%arg1) : (tensor<i1>) -> tensor<f32>
  // CHECK-NOT: "tf.StackPushV2"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: "tf.AssignVariableOp"(%[[TARG0:.*]], %[[UPDATE]])
  // CHECK: "tf.AssignVariableOp"(%[[EARG1:.*]],
  // CHECK-NOT: "tf.StackPushV2"
  %push = "tf.StackPushV2"(%arg0, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}

// -----

// Tests PartitionedCall op with no signature change on callee.

// CHECK-LABEL: func @main
func @main() -> () {
  "tf.PartitionedCall"() {f = @callee, config = "", config_proto = "", executor_type = ""} : () -> ()
  return
}
// CHECK: func @callee()
func @callee() -> () attributes {sym_visibility = "public"} {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.Stack
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  %push = "tf.StackPushV2"(%stack, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Tests that the pass reports error on unknown stack size.

func @main(%arg0: tensor<i32>) -> tensor<2xi32> {
  // expected-error @+1 {{unknown max element count}}
  %stack = "tf.StackV2"(%arg0) {elem_type = i32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %elem = "tf._SomeOp"() : () -> tensor<2xi32>
  %push = "tf.StackPushV2"(%stack, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<2xi32>) -> tensor<2xi32>
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  return %push : tensor<2xi32>
}

// -----

// Tests that the pass reports error on unknown element shape.

func @main(%arg0: tensor<i32>)  -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{cannot infer element shape of stack}}
  %stack = "tf.StackV2"(%max_size) {elem_type = i32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %elem = "tf._SomeOp"() : () -> tensor<*xi32>
  %push = "tf.StackPushV2"(%stack, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<*xi32>) -> tensor<*xi32>
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  return
}

// -----

// Tests that the pass reports error on ambiguous stack.

func @main(%arg0: tensor<i1>) -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %stack = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %stack2 = "tf.StackV2"(%max_size) {elem_type = f32, stack_name = "s2"} : (tensor<i32>) -> tensor<!tf.resource>
  %if_op = "tf.If"(%arg0, %stack, %stack2) {then_branch = @if_then, else_branch = @if_else, is_stateless = false}
    : (tensor<i1>, tensor<!tf.resource>, tensor<!tf.resource>) -> tensor<!tf.resource>
  // expected-error @+1 {{unknown stack}}
  %pop = "tf.StackPopV2"(%if_op) : (tensor<!tf.resource>) -> tensor<f32>
  "tf.StackCloseV2"(%stack) : (tensor<!tf.resource>) -> ()
  // CHECK: return
  return
}
func @if_then(%arg0: tensor<!tf.resource>, %arg1: tensor<!tf.resource>) -> tensor<!tf.resource> {
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  %push = "tf.StackPushV2"(%arg0, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}
func @if_else(%arg0: tensor<!tf.resource>, %arg1: tensor<!tf.resource>) -> tensor<!tf.resource> {
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  %push = "tf.StackPushV2"(%arg1, %elem) {swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
  return %arg1 : tensor<!tf.resource>
}
