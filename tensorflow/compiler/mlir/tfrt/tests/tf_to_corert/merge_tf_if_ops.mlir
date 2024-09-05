// RUN: tf-tfrt-opt -tfrt-merge-tf-if-ops %s | FileCheck %s -dump-input=fail

func.func @no_side_effect_then_0(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %0 : tensor<i32>, tensor<i32>
}

func.func @no_side_effect_else_0(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%x, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%y, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

func.func @no_side_effect_then_1(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @no_side_effect_else_1(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%x, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%y, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

func.func @nested_if_op_then_0(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %0 : tensor<i32>, tensor<i32>
}

func.func @nested_if_op_else_0(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

func.func @nested_if_op_then_1(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @nested_if_op_else_1(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func private @merge_stateless_merged_if_0_0_then
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK: [[r0:%.*]] = "tf.AddV2"([[x]], [[y]])
// CHECK: return [[r0]], [[r0]], [[r0]]

// CHECK-LABEL: func private @merge_stateless_merged_if_0_0_else
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK-DAG: [[cst:%.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}>
// CHECK-DAG: [[cst_0:%.*]] = "tf.Const"() <{value = dense<2> : tensor<i32>}>
// CHECK: [[r0:%.*]] = "tf.AddV2"([[x]], [[cst]])
// CHECK: [[r1:%.*]] = "tf.AddV2"([[y]], [[r0]])
// CHECK: [[r2:%.*]] = "tf.AddV2"([[x]], [[cst_0]])
// CHECK: [[r3:%.*]] = "tf.AddV2"([[y]], [[r2]])
// CHECK: return [[r0]], [[r1]], [[r3]]

// CHECK-LABEL: func @merge_stateless
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>)
func.func @merge_stateless(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: [[res:%.*]]:3 = "tf.If"([[cond]], [[x]], [[y]])
  // CHECK-SAME: <{else_branch = @merge_stateless_merged_if_0_0_else, is_stateless = true, then_branch = @merge_stateless_merged_if_0_0_then}>
  // CHECK-SAME: (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK-NEXT: return [[res]]#0, [[res]]#1, [[res]]#2
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @merge_nested_if_op_merged_if_0_0_then
// CHECK-SAME: ([[cond:%.*]]: tensor<i1>, [[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK-NEXT: [[r0:%.*]] = "tf.AddV2"([[x]], [[y]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT: return [[r0]], [[r0]], [[r0]]

// CHECK-LABEL: func private @merge_nested_if_op_merged_if_0_0_else_merged_if_1_0_then
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK-NEXT: [[r0:%.*]] = "tf.AddV2"([[x]], [[y]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT: return [[r0]], [[r0]], [[r0]]

// CHECK-LABEL: func private @merge_nested_if_op_merged_if_0_0_else_merged_if_1_0_else
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK-NEXT: [[cst:%.*]] = "tf.Const"() <{value = dense<2> : tensor<i32>}>
// CHECK-NEXT: [[cst_0:%.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}>
// CHECK-NEXT: [[r0:%.*]] = "tf.AddV2"([[x]], [[cst_0]])
// CHECK-NEXT: [[r1:%.*]] = "tf.AddV2"([[y]], [[r0]])
// CHECK-NEXT: [[r2:%.*]] = "tf.AddV2"([[x]], [[cst]])
// CHECK-NEXT: [[r3:%.*]] = "tf.AddV2"([[y]], [[r2]])
// CHECK-NEXT: return [[r0]], [[r1]], [[r3]]

// CHECK-LABEL: func private @merge_nested_if_op_merged_if_0_0_else
// CHECK-SAME: ([[cond:%.*]]: tensor<i1>, [[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK-NEXT: [[r0:%.*]]:3 = "tf.If"(%arg0, %arg1, %arg2) <{else_branch = @merge_nested_if_op_merged_if_0_0_else_merged_if_1_0_else, is_stateless = true, then_branch = @merge_nested_if_op_merged_if_0_0_else_merged_if_1_0_then}> : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
// CHECK-NEXT: return [[r0]]#0, [[r0]]#1, [[r0]]#2

// CHECK-LABEL: func @merge_nested_if_op
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>, [[nested_cond:%.*]]: tensor<i1>)
func.func @merge_nested_if_op(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>, %nested_cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: [[res:%.*]]:3 = "tf.If"([[cond]], [[nested_cond]], [[x]], [[y]])
  // CHECK-SAME: <{else_branch = @merge_nested_if_op_merged_if_0_0_else, is_stateless = true, then_branch = @merge_nested_if_op_merged_if_0_0_then}>
  // CHECK-SAME: (tensor<i1>, tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK-NEXT: return [[res]]#0, [[res]]#1, [[res]]#2
  %0, %1 = "tf.If"(%cond, %nested_cond, %x, %y) {else_branch = @nested_if_op_else_0, then_branch = @nested_if_op_then_0, is_stateless = true} : (tensor<i1>, tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %nested_cond, %x, %y) {else_branch = @nested_if_op_else_1, then_branch = @nested_if_op_then_1, is_stateless = true} : (tensor<i1>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @merge_side_effect
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>)
func.func @merge_side_effect(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: tf.If
  // CHECK-SAME: is_stateless = false
  // CHECK-NEXT: return
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @multiple_uses
func.func @multiple_uses(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: tf.If
  // CHECK-SAME: <{else_branch = @multiple_uses_merged_if_0_0_else, is_stateless = true, then_branch = @multiple_uses_merged_if_0_0_then}>
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}
