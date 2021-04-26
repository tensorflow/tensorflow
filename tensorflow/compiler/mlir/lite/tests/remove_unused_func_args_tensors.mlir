// RUN: tf-opt %s -split-input-file -tfl-remove-unused-function-args --cse | FileCheck %s

// Test for case with no global tensor references.
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "Variable", type = tensor<1x10xf32>, value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> ()
  func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @Variable}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %1 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    return %1 : tensor<1x10xf32>
  }

  // CHECK: func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
}

// -----

// Test for case with used global tensor references.
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "Variable", type = tensor<1x10xf32>, value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "UnusedVariable", type = tensor<1x10xf32>, value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> ()
  func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @Variable},
  %arg2: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @UnusedVariable}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
  }

  // CHECK: func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @Variable}) -> (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK-NEXT: "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
}
