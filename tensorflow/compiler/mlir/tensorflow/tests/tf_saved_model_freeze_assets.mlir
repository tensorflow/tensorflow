// RUN: tf-opt -verify-diagnostics -tf-saved-model-freeze-assets -split-input-file %s | FileCheck %s

module attributes {tf_saved_model.semantics} {

  // Test case: Basic freezing.

  "tf_saved_model.asset"() {filename = "assets/table.txt", sym_name = "v"} : () -> ()

  // CHECK: func @f()
  func @f(%arg0: tensor<!tf.string> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
    "tf.InitializeTableFromTextFileV2"(%0, %arg0) {delimiter = "\09", device = "", key_index = -2 : i64, offset = 0 : i64, value_index = -1 : i64, vocab_size = 437 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
    // CHECK: [[CST:%.+]] = "tf.Const"() {value = dense<"assets/table.txt"> : tensor<1x!tf.string>} : () -> tensor<1x!tf.string>
    // CHECK: [[HASHTABLE:%.+]] = "tf.HashTableV2"()
    // CHECK: "tf.InitializeTableFromTextFileV2"([[HASHTABLE]], [[CST]])
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non-bound inputs.
  // The pass shouldn't do anything in this case.

  // CHECK: func @f(%arg0
  func @f(%arg0: tensor<!tf.string> {tf_saved_model.index_path = [0]})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
    "tf.InitializeTableFromTextFileV2"(%0, %arg0) {delimiter = "\09", device = "", key_index = -2 : i64, offset = 0 : i64, value_index = -1 : i64, vocab_size = 437 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
    // CHECK: "tf.InitializeTableFromTextFileV2"(%0, %arg0)
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non tf.InitializeTableFromTextFileV2 op usages.

  "tf_saved_model.asset"() {filename = "assets/table.txt", sym_name = "v"} : () -> ()

  // CHECK: func @f(%arg0
  func @f(%arg0: tensor<!tf.string> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.string>) -> ()
    return
  }

  func private @f_callee(%arg0: tensor<!tf.string>) {
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.asset"() {filename = "assets/table.txt", sym_name = "v"} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/table2.txt", sym_name = "w"} : () -> ()

  // CHECK: func @f()
  func @f(%arg0: tensor<!tf.string> {tf_saved_model.bound_input = @v}, %arg1: tensor<!tf.string> {tf_saved_model.bound_input = @w})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
    "tf.InitializeTableFromTextFileV2"(%0, %arg0) {delimiter = "\09", device = "", key_index = -2 : i64, offset = 0 : i64, value_index = -1 : i64, vocab_size = 437 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
    %1 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
    "tf.InitializeTableFromTextFileV2"(%1, %arg1) {delimiter = "\09", device = "", key_index = -2 : i64, offset = 0 : i64, value_index = -1 : i64, vocab_size = 437 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
    // CHECK: [[CST_1:%.+]] = "tf.Const"() {value = dense<"assets/table2.txt"> : tensor<1x!tf.string>} : () -> tensor<1x!tf.string>
    // CHECK: [[CST:%.+]] = "tf.Const"() {value = dense<"assets/table.txt"> : tensor<1x!tf.string>} : () -> tensor<1x!tf.string>
    // CHECK: [[HASHTABLE:%.+]] = "tf.HashTableV2"()
    // CHECK: "tf.InitializeTableFromTextFileV2"([[HASHTABLE]], [[CST]])
    // CHECK: [[HASHTABLE_1:%.+]] = "tf.HashTableV2"()
    // CHECK: "tf.InitializeTableFromTextFileV2"([[HASHTABLE_1]], [[CST_1]])
    return
  }
}

// -----

// Test running the pass on a module that does not have
// tf_saved_model.semantics.
module {}
