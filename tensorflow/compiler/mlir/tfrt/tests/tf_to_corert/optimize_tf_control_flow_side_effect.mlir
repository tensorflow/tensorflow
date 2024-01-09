// RUN: tf-tfrt-opt -split-input-file -tfrt-optimize-tf-control-flow-side-effect %s | FileCheck %s

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

// CHECK-LABEL: func @nested_set_stateless
func.func @nested_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = true
  %1 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @no_side_effect_nested_body, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0 = "tf.While"(%arg) { cond = @no_side_effect_cond, body = @no_side_effect_nested_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @no_side_effect_nested_body
func.func @no_side_effect_nested_body(%arg: tensor<i32>) -> tensor<i32> {
  %const = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %cond = "tf.Less"(%arg, %const) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = true
  %0 = "tf.If"(%cond, %arg) {else_branch = @no_side_effect_body, then_branch = @no_side_effect_body2, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %1 = "tf.While"(%0) { cond = @no_side_effect_cond, body = @no_side_effect_body, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  func.return %1 : tensor<i32>
}

// -----

func.func @no_side_effect_cond(%arg: tensor<i32>, %handle: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Less"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

func.func @side_effect_body(%arg: tensor<i32>, %handle: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) {
  %0 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1 = "tf.AddV2"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %handle : tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>
}

func.func @no_side_effect_body(%arg: tensor<i32>, %handle: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) {
  %0 = "tf.AddV2"(%arg, %arg) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %handle : tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>
}

// CHECK-LABEL: func @no_set_stateless
func.func @no_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  %handle = "tf.VarHandleOp"() { container = "", shared_name = "var" } : () -> tensor<!tf_type.resource<tensor<i32>>>
  "tf.AssignVariableOp"(%handle, %arg) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0, %h1 = "tf.While"(%arg, %handle) { cond = @no_side_effect_cond, body = @side_effect_body, is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1, %h2 = "tf.If"(%cond, %arg, %handle) {else_branch = @no_side_effect_body, then_branch = @side_effect_body, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}


// CHECK-LABEL: func @side_effect_nested_body
func.func @side_effect_nested_body(%arg: tensor<i32>, %handle: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) {
  %const = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0, %h1 = "tf.While"(%arg, %handle) { cond = @no_side_effect_cond, body = @side_effect_body, is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  %cond = "tf.Less"(%0, %const) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1, %h2 = "tf.If"(%cond, %0, %handle) {else_branch = @no_side_effect_body, then_branch = @side_effect_body, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  func.return %1, %handle : tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>
}

// CHECK-LABEL: func @nested_no_set_stateless
func.func @nested_no_set_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>, tensor<i32>) {
  %handle = "tf.VarHandleOp"() { container = "", shared_name = "var" } : () -> tensor<!tf_type.resource<tensor<i32>>>
  "tf.AssignVariableOp"(%handle, %arg) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = false
  %0, %h1 = "tf.While"(%arg, %handle) { cond = @no_side_effect_cond, body = @side_effect_nested_body, is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  // CHECK: tf.If
  // CHECK-SAME: is_stateless = false
  %1, %h2 = "tf.If"(%cond, %arg, %handle) {else_branch = @no_side_effect_body, then_branch = @side_effect_nested_body, is_stateless = false} : (tensor<i1>, tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

// Set stateless if the body contains only read-only side-effecting ops.

func.func private @cond(%arg: tensor<i32>, %handle: tensor<!tf_type.resource>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Less"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

func.func private @body(%arg: tensor<i32>, %handle: tensor<!tf_type.resource>) -> (tensor<i32>, tensor<!tf_type.resource>) {
  %default = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.LookupTableFindV2"(%handle, %arg, %default) : (tensor<!tf_type.resource>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %handle : tensor<i32>, tensor<!tf_type.resource>
}

// CHECK-LABEL: func @set_readonly_stateless
func.func @set_readonly_stateless(%arg: tensor<i32>, %cond: tensor<i1>) -> tensor<i32> {
  %handle = "tf.HashTableV2"() {container = "", key_dtype = i32, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i32} : () -> tensor<!tf_type.resource>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0, %handle_2 = "tf.While"(%arg, %handle) { cond = @cond, body = @body, is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<i32>, tensor<!tf_type.resource>)
  func.return %0: tensor<i32>
}

// -----

// Set stateless if write side-effecting ops are inside the initializer.

module attributes {tf_saved_model.semantics} {

"tf_saved_model.session_initializer"() {initializers = [@init]} : () -> ()

func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init"]} {
  %keys = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32> } : () -> tensor<4xi32>
  %values = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32> } : () -> tensor<4xi32>
  %handle = "tf.HashTableV2"() {container = "", key_dtype = i32, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i32} : () -> tensor<!tf_type.resource>
  "tf.LookupTableImportV2"(%handle, %keys, %values) : (tensor<!tf_type.resource>, tensor<4xi32>, tensor<4xi32>) -> ()
  func.return
}

func.func private @cond(%arg: tensor<i32>, %handle: tensor<!tf_type.resource>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Less"(%arg, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

func.func private @body(%arg: tensor<i32>, %handle: tensor<!tf_type.resource>) -> (tensor<i32>, tensor<!tf_type.resource>) {
  %default = "tf.Const"() {value = dense<16> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.LookupTableFindV2"(%handle, %arg, %default) : (tensor<!tf_type.resource>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %handle : tensor<i32>, tensor<!tf_type.resource>
}

// CHECK-LABEL: func @set_readonly_stateless
func.func @set_readonly_stateless(%arg: tensor<i32> {tf_saved_model.index_path = ["arg"]}, %cond: tensor<i1> {tf_saved_model.index_path = ["cond"]}) -> (tensor<i32> {tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["main"]} {
  %handle = "tf.HashTableV2"() {container = "", key_dtype = i32, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i32} : () -> tensor<!tf_type.resource>
  // CHECK: tf.While
  // CHECK-SAME: is_stateless = true
  %0, %handle_2 = "tf.While"(%arg, %handle) { cond = @cond, body = @body, is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<i32>, tensor<!tf_type.resource>)
  func.return %0: tensor<i32>
}

}
