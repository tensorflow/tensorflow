// RUN: tf-tfrt-opt -tfrt-merge-tf-if-ops %s | FileCheck %s -dump-input=fail

func @no_side_effect_then_0(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0, %0 : tensor<i32>, tensor<i32>
}

func @no_side_effect_else_0(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%x, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%y, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1, %2 : tensor<i32>, tensor<i32>
}

func @no_side_effect_then_1(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.AddV2"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

func @no_side_effect_else_1(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%x, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%y, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}

// CHECK-LABEL: func private @merge_stateless_merged_if_0_then
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK: [[r0:%.*]]:2 = "tf.PartitionedCall"
// CHECK-SAME: f = @no_side_effect_then_0
// CHECK: [[r1:%.*]] = "tf.PartitionedCall"
// CHECK-SAME: f = @no_side_effect_then_1
// CHECK: return [[r0]]#0, [[r0]]#1, [[r1]]

// CHECK-LABEL: func private @merge_stateless_merged_if_0_else
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>)
// CHECK: [[r0:%.*]]:2 = "tf.PartitionedCall"
// CHECK-SAME: f = @no_side_effect_else_0
// CHECK: [[r1:%.*]] = "tf.PartitionedCall"
// CHECK-SAME: f = @no_side_effect_else_1
// CHECK: return [[r0]]#0, [[r0]]#1, [[r1]]

// CHECK-LABEL: func @merge_stateless
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>)
func @merge_stateless(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: [[res:%.*]]:3 = "tf.If"([[cond]], [[x]], [[y]])
  // CHECK-SAME: {else_branch = @merge_stateless_merged_if_0_else, is_stateless = true, then_branch = @merge_stateless_merged_if_0_then}
  // CHECK-SAME: (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK-NEXT: return [[res]]#0, [[res]]#1, [[res]]#2
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @not_merge_side_effect
// CHECK-SAME: ([[x:%.*]]: tensor<i32>, [[y:%.*]]: tensor<i32>, [[cond:%.*]]: tensor<i1>)
func @not_merge_side_effect(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: tf.If
  // CHECK-NEXT: tf.If
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @multiple_uses
func @multiple_uses(%x: tensor<i32>, %y: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: tf.If
  // CHECK-SAME: {else_branch = @multiple_uses_merged_if_0_else, is_stateless = true, then_branch = @multiple_uses_merged_if_0_then}
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_0, then_branch = @no_side_effect_then_0, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %2 = "tf.If"(%cond, %x, %y) {else_branch = @no_side_effect_else_1, then_branch = @no_side_effect_then_1, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}
