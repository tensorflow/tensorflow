// RUN: tf-opt %s -split-input-file -tfl-legalize-variables-tf --cse | FileCheck %s

// Test for case with no session initialize op.
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: serving_default
  func.func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}, tensor<1x10xi64> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    %handle_1 = "tf.VarHandleOp"() {container="c", shared_name="b"} : () -> tensor<!tf_type.resource<tensor<1x10xi64>>>
    %0 = "tf.ReadVariableOp"(%handle_0) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    "tf.AssignVariableOp"(%handle_0, %1) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    %2 = "tf.ReadVariableOp"(%handle_0) {device = ""} : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    %3 = "tf.ReadVariableOp"(%handle_1) {device = ""} : (tensor<!tf_type.resource<tensor<1x10xi64>>>) -> tensor<1x10xi64>
    func.return %2, %3 : tensor<1x10xf32>, tensor<1x10xi64>
  }

  // CHECK: %[[RESOURCE:.*]] = "tfl.var_handle"() {container = "c", shared_name = "a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
  // CHECK: %[[RESOURCE_1:.*]] = "tfl.var_handle"() {container = "c", shared_name = "b"} : () -> tensor<!tf_type.resource<tensor<1x10xi64>>>
  // CHECK: %[[VAR_VAL:.*]] = "tfl.read_variable"(%[[RESOURCE]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
  // CHECK: %[[ADD:.*]] = tfl.add %[[VAR_VAL]], %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
  // CHECK: "tfl.assign_variable"(%[[RESOURCE]], %[[ADD]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
  // CHECK: %[[RESULT:.*]] = "tfl.read_variable"(%[[RESOURCE]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
  // CHECK: %[[VAR:.*]] = "tfl.read_variable"(%[[RESOURCE_1]]) : (tensor<!tf_type.resource<tensor<1x10xi64>>>) -> tensor<1x10xi64>
  // CHECK:  return %[[RESULT]], %[[VAR]] : tensor<1x10xf32>, tensor<1x10xi64>
}

// -----

// Test for case with existing session initialize op.
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: init_all_tables
  func.func @init_all_tables()
  attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %cst_0 = arith.constant dense<["a", "b", "c", "d"]> : tensor<4x!tf_type.string>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_dba2ccaa-f1b1-46d6-b276-98008f69da71", use_node_name_sharing = false, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf_type.resource>, tensor<4xi64>, tensor<4x!tf_type.string>) -> ()
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    %cst_1 = arith.constant dense<1.0> : tensor<1x10xf32>
    "tf.AssignVariableOp"(%handle_0, %cst_1) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    func.return
    // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<1x10xf32>
    // CHECK: %[[RESOURCE:.*]] = "tfl.var_handle"() {container = "c", shared_name = "a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    // CHECK: "tfl.assign_variable"(%[[RESOURCE]], %[[CST]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
  }

  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"() <{initializers = [@init_all_tables]}> : () -> ()

  // CHECK-LABEL: serving_default
  func.func @serving_default(%arg0: tensor<1x10xf32> {tf_saved_model.index_path = ["x"]}) ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    %0 = "tf.ReadVariableOp"(%handle_0) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    "tf.AssignVariableOp"(%handle_0, %1) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    %2 = "tf.ReadVariableOp"(%handle_0) {device = ""} : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    func.return %2 : tensor<1x10xf32>
    // CHECK: %[[RESOURCE:.*]] = "tfl.var_handle"() {container = "c", shared_name = "a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    // CHECK: %[[VAR_VAL:.*]] = "tfl.read_variable"(%[[RESOURCE]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    // CHECK: %[[ADD:.*]] = tfl.add %[[VAR_VAL]], %arg0 {fused_activation_function = "NONE"} : tensor<1x10xf32>
    // CHECK: "tfl.assign_variable"(%[[RESOURCE]], %[[ADD]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
    // CHECK: %[[RESULT:.*]] = "tfl.read_variable"(%[[RESOURCE]]) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    // CHECK:  return %[[RESULT]] : tensor<1x10xf32>
  }
}

// -----

// Don't legalize if type is not supported.
module attributes {tf_saved_model.semantics} {
  func.func @serving_default() ->
    (tensor<1x10xui64> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xui64>>>
    %0 = "tf.ReadVariableOp"(%handle_0) {device = ""} : (tensor<!tf_type.resource<tensor<1x10xui64>>>) -> tensor<1x10xui64>
    func.return %0 : tensor<1x10xui64>
  }

  // CHECK: tf.ReadVariableOp
}

// -----

// Don't legalize if type is not supported.
module attributes {tf_saved_model.semantics, tfl._legalize_tfl_variables = true} {
  func.func @serving_default() ->
    () attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = ""}, tf_saved_model.exported_names = ["serving_default"]} {
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xui64>>>
    %cst = arith.constant dense<2> : tensor<1x10xui64>
    "tf.AssignVariableOp"(%handle_0, %cst) : (tensor<!tf_type.resource<tensor<1x10xui64>>>, tensor<1x10xui64>) -> ()
    func.return
  }

  // CHECK: tf.VarHandleOp
  // CHECK: tf.AssignVariableOp
}

// -----

// Don't legalize if disabled by earlier analysis.
module attributes {tf_saved_model.semantics, tfl._legalize_tfl_variables = false} {
  func.func @serving_default() ->
    (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %handle_0 = "tf.VarHandleOp"() {container="c", shared_name="a"} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
    %0 = "tf.ReadVariableOp"(%handle_0) : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
    func.return %0 : tensor<1x10xf32>
  }

  // CHECK: tf.ReadVariableOp
}
