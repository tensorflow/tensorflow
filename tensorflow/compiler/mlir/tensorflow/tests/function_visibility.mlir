// RUN: tf-opt -tf-saved-model-mark-func-visibility -split-input-file %s | FileCheck --check-prefix=SAVEDMODEL %s --dump-input=fail
// RUN: tf-opt -tf-mark-func-visibility -split-input-file -verify-diagnostics %s | FileCheck %s --dump-input=fail


module attributes {tf_saved_model.semantics} {
  // SAVEDMODEL: func @func_exported_1() attributes {tf_saved_model.exported_names = ["func_exported_1"]}
  func @func_exported_1() attributes {tf_saved_model.exported_names = ["func_exported_1"]} {
    "tf.some_call"() {callee = {callee = {callee = @child}}} : () -> ()
    return
  }

  // SAVEDMODEL: func @func_exported_2() attributes {tf_saved_model.exported_names = ["func_exported_2"]}
  func @func_exported_2() attributes {tf_saved_model.exported_names = ["func_exported_2"]} {
    "tf.some_call"() {callee = {callee = {callee = @child}}} : () -> ()
    return
  }

  // SAVEDMODEL: func @func_not_exported() attributes {sym_visibility = "private"}
  func @func_not_exported() {
    return
  }

}

// -----

module {
  // CHECK: func @func_with_entry_spec(%arg0: tensor<1xi32>) -> tensor<1xi32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}}
  func @func_with_entry_spec(%arg0: tensor<1xi32>) -> tensor<1xi32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}} {
    return %arg0 : tensor<1xi32>
  }

  // CHECK: func @func_without_entry_spec(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> attributes {sym_visibility = "private"}
  func @func_without_entry_spec(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.AddV2"(%arg0, %arg1) {T = i32, device = ""} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    return %0 : tensor<*xi32>
  }
}

// -----

module {
  // expected-error @+1 {{can't overwrite the visibility of function private_func_with_entry_spec with private visibility}}
  func @private_func_with_entry_spec(%arg0: tensor<1xi32>) -> tensor<1xi32> attributes {tf.entry_function = {inputs = "x", outputs = "y"}, sym_visibility = "private"} {
    return %arg0 : tensor<1xi32>
  }
}
