// RUN: tf-opt -split-input-file -tfl-insert-call-once-op %s | FileCheck %s

// Tests that new call_once op is added when there is a session initializer.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()

  func @init_all_tables()
  attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    %cst = constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %cst_0 = constant dense<["a", "b", "c", "d"]> : tensor<4x!tf.string>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_dba2ccaa-f1b1-46d6-b276-98008f69da71", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf.resource>, tensor<4xi64>, tensor<4x!tf.string>) -> ()
    return
    // CHECK-LABEL: @init_all_tables
  }

  func @serving_default(%arg0: tensor<i64> {tf_saved_model.index_path = ["x"]}) -> (tensor<*x!tf.string> {tf_saved_model.index_path = ["r"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "hash_table_Lookup/LookupTableFindV2:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = constant dense<"f"> : tensor<!tf.string>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_dba2ccaa-f1b1-46d6-b276-98008f69da71", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
    %1 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf.resource>, tensor<i64>, tensor<!tf.string>) -> tensor<*x!tf.string>
    return %1 : tensor<*x!tf.string>
    // CHECK-LABEL: @serving_default
    // CHECK: "tfl.call_once"() {session_init_function = "init_all_tables"} : () -> ()
  }
}

// -----

// Tests that no call_once op is added.

module attributes {tf_saved_model.semantics} {
  func @no_call_once(%arg0: tensor<i64> {tf_saved_model.index_path = ["x"]}) -> (tensor<i64> {tf_saved_model.index_path = ["r"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    return %arg0 : tensor<i64>
    // CHECK-LABEL: no_call_once
    // CHECK-NOT: "tfl.call_once"
  }
}
