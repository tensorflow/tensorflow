// RUN: tf-opt -tf-init-text-file-to-import-test %s | FileCheck %s

// Tests that the tf.InitializeTableFromTextFileV2 op are inlined.

func @init_all_tables() {
  %cst = constant dense<"%FILE_PLACEHOLDER"> : tensor<!tf.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = -1 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
  return
  // CHECK-LABEL: func @init_all_tables
  // CHECK: [[CST:%.*]]  = constant dense<["apple", "banana", "grape"]> : tensor<3x!tf.string>
  // CHECK: [[CST_0:%.*]]  = constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: "tf.LookupTableImportV2"([[VAL]], [[CST]], [[CST_0]])
}

// Tests that the tf.InitializeTableFromTextFileV2 op with explicit vocab size.

func @init_all_tables_with_explicit_vocab_size() {
  %cst = constant dense<"%FILE_PLACEHOLDER"> : tensor<!tf.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = 2 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
  return
  // CHECK-LABEL: func @init_all_tables_with_explicit_vocab_size
  // CHECK: [[CST:%.*]]  = constant dense<["apple", "banana"]> : tensor<2x!tf.string>
  // CHECK: [[CST_0:%.*]]  = constant dense<[0, 1]> : tensor<2xi64>
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: "tf.LookupTableImportV2"([[VAL]], [[CST]], [[CST_0]])
}
