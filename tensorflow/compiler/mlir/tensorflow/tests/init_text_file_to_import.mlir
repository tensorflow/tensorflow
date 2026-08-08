// Copyright 2026 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-opt -tf-init-text-file-to-import-test %s | FileCheck %s

// Tests that the tf.InitializeTableFromTextFileV2 op are inlined.

func.func @init_all_tables() {
  %cst = arith.constant dense<"%FILE_PLACEHOLDER"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  func.return
  // CHECK-LABEL: func @init_all_tables
  // CHECK-DAG: [[CST:%.*]]  = arith.constant dense<["apple", "banana", "grape"]> : tensor<3x!tf_type.string>
  // CHECK-DAG: [[CST_0:%.*]]  = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: "tf.LookupTableImportV2"([[VAL]], [[CST]], [[CST_0]])
}

// Tests that the tf.InitializeTableFromTextFileV2 op with explicit vocab size.

func.func @init_all_tables_with_explicit_vocab_size() {
  %cst = arith.constant dense<"%FILE_PLACEHOLDER"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_/tmp/vocab.txt_-2_-1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.InitializeTableFromTextFileV2"(%0, %cst) {delimiter = " ", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = 2 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  func.return
  // CHECK-LABEL: func @init_all_tables_with_explicit_vocab_size
  // CHECK-DAG: [[CST:%.*]]  = arith.constant dense<["apple", "banana"]> : tensor<2x!tf_type.string>
  // CHECK-DAG: [[CST_0:%.*]]  = arith.constant dense<[0, 1]> : tensor<2xi64>
  // CHECK: [[VAL:%.*]] = "tf.HashTableV2"()
  // CHECK: "tf.LookupTableImportV2"([[VAL]], [[CST]], [[CST_0]])
}
