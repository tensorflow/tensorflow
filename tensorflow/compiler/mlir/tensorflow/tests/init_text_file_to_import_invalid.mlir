// RUN: tf-opt -split-input-file -verify-diagnostics -tf-init-text-file-to-import %s | FileCheck %s

// Tests that the given vocabulary file does not exist.

func.func @init_all_tables() {
  %cst = arith.constant dense<"vocab_file_does_not_exist.txt"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  // expected-error @+1 {{'tf.InitializeTableFromTextFileV2' op failed to open vocabulary file (vocab_file_does_not_exist.txt): cannot open input file 'vocab_file_does_not_exist.txt': No such file or directory}}
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  func.return
}

// -----

// Tests that the tf.InitializeTableFromTextFileV2 op is not converted since
// unsupported key_index, -1.

func.func @init_all_tables() {
  %cst = arith.constant dense<"vocab_file_does_not_exist.txt"> : tensor<!tf_type.string> %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -1 : i64, value_index = -1 : i64, vocab_size = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  func.return
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: tf.InitializeTableFromTextFileV2"
}

// -----

// Tests that the tf.InitializeTableFromTextFileV2 op is not converted since
// unsupported value_index, 0.

func.func @init_all_tables() {
  %cst = arith.constant dense<"vocab_file_does_not_exist.txt"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = 0 : i64, vocab_size = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  func.return
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: tf.InitializeTableFromTextFileV2"
}
