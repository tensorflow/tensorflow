// RUN: tf-quant-opt %s -tf-quant-merge-save-function-ops-to-main \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// Test that the @tf_quant_save's ops are cloned to @main.

module attributes {tf_saved_model.semantics} {
  func.func private @tf_quant__save(%arg: tensor<!tf_type.string>) -> () {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
      %out_0, %ctl_0 = tf_executor.island wraps "tf.ReadVariableOp"(%out) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
      %out_1, %ctl_1 = tf_executor.island wraps "tf.Const"() {value = dense<"var_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_2, %ctl_2 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %ctl_3 = tf_executor.island wraps "tf.SaveV2"(%arg, %out_1, %out_2, %out_0) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xf32>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }

  func.func @main(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}) -> ()
      attributes {tf.entry_function = {inputs = "tf_file_prefix:0", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
}
// Save function should be erased.
// CHECK-NOT: @tf_quant__save

// Test that the contents of @tf_quant__save are copied to @main.
// CHECK: func.func @main
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}
// CHECK: tf_executor.graph
// CHECK: %[[VAR_HANDLE:.*]], {{.*}} = tf_executor.island wraps "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK: %[[READ_VARIABLE:.*]], {{.*}} = tf_executor.island wraps "tf.ReadVariableOp"(%[[VAR_HANDLE]])
// CHECK-DAG: %[[CST_0:.*]], {{.*}} = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<"var_0"> : tensor<1x!tf_type\.string>.*}}}>
// CHECK-DAG: %[[CST_1:.*]], {{.*}} = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<""> : tensor<1x!tf_type\.string>.*}}}>
// CHECK: %[[CTL_0:.*]] = tf_executor.island wraps "tf.SaveV2"(%[[ARG_0]], %[[CST_0]], %[[CST_1]], %[[READ_VARIABLE]]) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xf32>) -> ()

// Test that the Identity op has been created to fetch the file prefix
// argument. It should also have control dependency to the `SaveV2` op.
// CHECK: %[[IDENTITY:.*]], %[[CTL_1:.*]] = tf_executor.island(%[[CTL_0]]) wraps "tf.Identity"(%[[ARG_0]])
// CHECK: tf_executor.fetch %[[CTL_1]] : !tf_executor.control
// CHECK: return

// -----

// Test that no ops are added to @main when @tf_quant__save function does
// not exist.

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}) -> ()
      attributes {tf.entry_function = {inputs = "tf_file_prefix:0", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
}
// CHECK: func.func @main
// CHECK: tf_executor.graph
// CHECK-NEXT: tf_executor.fetch

// -----

// Test error when @main op doesn't exist.

// expected-error @+1 {{Main function op not found.}}
module attributes {tf_saved_model.semantics} {
}

// -----

// Test that no ops are added to @main when there are no `GraphOp` in @main.

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}) -> ()
      attributes {tf.entry_function = {inputs = "tf_file_prefix:0", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    return
  }
// CHECK: func.func @main({{.*}}) attributes {{{.*}}} {
// CHECK-NEXT: return

  func.func private @tf_quant__save(%arg: tensor<!tf_type.string>) -> () {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {value = dense<"hello"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      tf_executor.fetch %ctl : !tf_executor.control
    }
    return
  }
}

// -----

// Test that no ops are added to @main when there are no `GraphOp` in
// @tf_quant__save.

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}) -> ()
      attributes {tf.entry_function = {inputs = "tf_file_prefix:0", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// CHECK: func.func @main({{.*}}) attributes {{{.*}}} {
// CHECK-NEXT: tf_executor.graph
// CHECK-NEXT: tf_executor.fetch

  func.func private @tf_quant__save(%arg: tensor<!tf_type.string>) -> () {
    return
  }
}

// -----

// Test that the @tf_quant_save's ops are cloned to @main. When there are no
// __tf_file_prefix argument in @main, confirm that it is created and wired
// to the newly created `IdentityOp`.

module attributes {tf_saved_model.semantics} {
  func.func private @tf_quant__save(%arg: tensor<!tf_type.string>) -> () {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
      %out_0, %ctl_0 = tf_executor.island wraps "tf.ReadVariableOp"(%out) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
      %out_1, %ctl_1 = tf_executor.island wraps "tf.Const"() {value = dense<"var_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_2, %ctl_2 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %ctl_3 = tf_executor.island wraps "tf.SaveV2"(%arg, %out_1, %out_2, %out_0) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xf32>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }

  func.func @main() -> () attributes {
      tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
}
// Save function should be erased.
// CHECK-NOT: @tf_quant__save

// Test that the contents of @tf_quant__save are copied to @main.
// CHECK: func.func @main
// Test that the "__tf_file_prefix" argument of type `tensor<!tf_type_string>`
// has been created.
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}
// CHECK-SAME: tf.entry_function = {inputs = "__tf_file_prefix:0", outputs = ""}
// CHECK: tf_executor.graph
// CHECK: %[[VAR_HANDLE:.*]], {{.*}} = tf_executor.island wraps "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK: %[[READ_VARIABLE:.*]], {{.*}} = tf_executor.island wraps "tf.ReadVariableOp"(%[[VAR_HANDLE]])
// CHECK-DAG: %[[CST_0:.*]], {{.*}} = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<"var_0"> : tensor<1x!tf_type\.string>.*}}}>
// CHECK-DAG: %[[CST_1:.*]], {{.*}} = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<""> : tensor<1x!tf_type\.string>.*}}}>
// CHECK: %[[CTL_0:.*]] = tf_executor.island wraps "tf.SaveV2"(%[[ARG_0]], %[[CST_0]], %[[CST_1]], %[[READ_VARIABLE]]) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xf32>) -> ()

// Test that the Identity op has been created to fetch the file prefix
// argument. It should also have control dependency to the `SaveV2` op.
// CHECK: %[[IDENTITY:.*]], %[[CTL_1:.*]] = tf_executor.island(%[[CTL_0]]) wraps "tf.Identity"(%[[ARG_0]])
// CHECK: tf_executor.fetch %[[CTL_1]] : !tf_executor.control
// CHECK: return
