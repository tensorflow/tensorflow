// RUN: tf-tfrt-opt -tfrt-optimize-tf-control-flow-side-effect %s | FileCheck %s

func.func @no_side_effect_cond(%arg: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Less"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

func.func @no_side_effect_body(%arg: tensor<i32>) -> tensor<i32> {
  %0 = "tf.AddV2"(%arg, %arg) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @no_side_effect_body2(%arg: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @set_stateless
func.func @set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @no_side_effect_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = true
  %1 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @no_side_effect_body2, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

func.func @side_effect_body(%arg: tensor<i32>) -> tensor<i32> {
  %handle = "tf.VarHandleOp"() { container = "", shared_name = "var" } : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1 = "tf.AddV2"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @no_set_stateless
func.func @no_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @side_effect_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @side_effect_body, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @no_side_effect_nested_body
func.func @no_side_effect_nested_body(%arg: tensor<i32>) -> tensor<i32> {
  %const = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @no_side_effect_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  %cond = "tf.Less"(%0, %const) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = true
  %1 = "tf.If"(%cond, %0) {else_branch = @no_side_effect_body, then_branch = @no_side_effect_body2, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @nested_set_stateless
func.func @nested_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @no_side_effect_nested_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = true
  %1 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @no_side_effect_nested_body, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @side_effect_nested_body
func.func @side_effect_nested_body(%arg: tensor<i32>) -> tensor<i32> {
  %const = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @side_effect_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  %cond = "tf.Less"(%0, %const) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1 = "tf.If"(%cond, %0) {else_branch = @no_side_effect_body, then_branch = @side_effect_body, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @nested_no_set_stateless
func.func @nested_no_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @side_effect_nested_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @side_effect_nested_body, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  func.return %0, %1 : tensor<i32>, tensor<i32>
}
