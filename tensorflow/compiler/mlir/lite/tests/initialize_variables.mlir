// RUN: tf-opt %s -split-input-file -tfl-initialize-variables-tf --cse | FileCheck %s

// Test for case with no session initialize op.
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "Variable", type = tensor<1x10xf32>, value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> ()
  func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @Variable}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    "tf.AssignVariableOp"(%arg1, %1) : (tensor<!tf.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    %2 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    return %2 : tensor<1x10xf32>
  }

  // CHECK: func @SessionInitializerFunction() attributes {tf_saved_model.exported_names = ["SessionInitializerFunction"]} {
  // CHECK: %[[RESOURCE:.*]] = "tfl.pseudo_const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[VAL:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK: "tfl.assign_variable"(%[[RESOURCE]], %[[VAL]]) : (tensor<1xi32>, tensor<1x10xf32>) -> ()
  // CHECK: return
  // CHECK: }
  // CHECK: "tf_saved_model.session_initializer"() {initializers = [@SessionInitializerFunction]} : () -> ()
  // CHECK: func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}
  // CHECK: "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
}

// -----

// Test for case with existing session initialize op.
module attributes {tf_saved_model.semantics} {
  func @init_all_tables()
  attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    %cst = constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %cst_0 = constant dense<["a", "b", "c", "d"]> : tensor<4x!tf.string>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_dba2ccaa-f1b1-46d6-b276-98008f69da71", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf.resource>, tensor<4xi64>, tensor<4x!tf.string>) -> ()
    return
    // CHECK-LABEL: @init_all_tables
    // CHECK: %[[RESOURCE:.*]] = "tfl.pseudo_const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    // CHECK: %[[VAL:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
    // CHECK: "tfl.assign_variable"(%[[RESOURCE]], %[[VAL]]) : (tensor<1xi32>, tensor<1x10xf32>) -> ()
  }

  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()

  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "Variable", type = tensor<1x10xf32>, value = dense<0.000000e+00> : tensor<1x10xf32>} : () -> ()
  func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<!tf.resource<tensor<1x10xf32>>> {tf_saved_model.bound_input = @Variable}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    "tf.AssignVariableOp"(%arg1, %1) : (tensor<!tf.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    %2 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    return %2 : tensor<1x10xf32>
    // CHECK: func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}
    // CHECK: "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
  }
}
