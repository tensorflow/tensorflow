// RUN: tf-opt %s -split-input-file -tfl-legalize-hashtables-tf --cse | FileCheck %s

// Test for case with string -> int64 hashtable.
func.func @hashtable_string_to_int64(%arg0: tensor<i64>) -> tensor<*xi64> {
  %cst = arith.constant dense<"f"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  %1 = "tf.LookupTableFindV2"(%0, %cst, %arg0) {device = ""} : (tensor<!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-LABEL: hashtable_string_to_int64
  // CHECK:       [[CST:%.*]] = arith.constant dense<"f"> : tensor<!tf_type.string>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf_type.string, table_id = 1530976467 : i32, value_dtype = i64} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], [[CST]], %arg0) : (tensor<1x!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-NEXT:  return [[FIND]] : tensor<*xi64>
  func.return %1 : tensor<*xi64>
}

// -----

// Test for case with int64 -> string hashtable.
func.func @hashtable_int64_to_string(%arg0: tensor<i64>) -> tensor<*x!tf_type.string> {
  %cst = arith.constant dense<"f"> : tensor<!tf_type.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
  %1 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf_type.resource>, tensor<i64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.string>
  // CHECK-LABEL: hashtable_int64_to_string
  // CHECK:       [[CST:%.*]] = arith.constant dense<"f"> : tensor<!tf_type.string>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = i64, table_id = 1530976467 : i32, value_dtype = !tf_type.string} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], %arg0, [[CST]]) : (tensor<1x!tf_type.resource>, tensor<i64>, tensor<!tf_type.string>) -> tensor<*x!tf_type.string>
  // CHECK-NEXT:  return [[FIND]] : tensor<*x!tf_type.string>
  func.return %1 : tensor<*x!tf_type.string>
}

// -----

// Test for case with unsupported string -> string mapping.
func.func @no_legalization_on_hashtable_string_to_string(%arg0: tensor<!tf_type.string>) {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
  // CHECK-LABEL: no_legalization_on_hashtable_string_to_string
  // CHECK-NEXT:  "tf.HashTableV2"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  func.return
}

// -----

// Test for case with import op.
func.func @hashtable_import(%arg0: tensor<5x!tf_type.string>) {
  %cst = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
  func.return
  // CHECK-LABEL: hashtable_import
  // CHECK:       [[CST:%.*]] = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf_type.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_0]]) : (tensor<1x!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
}

// -----

// Test for case with size op.
func.func @hashtable_size(%arg0: tensor<5x!tf_type.string>) -> tensor<i64> {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
  // CHECK-LABEL: hashtable_size
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf_type.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:  [[SIZE:%.*]] = "tfl.hashtable_size"([[HASH_TABLE]]) : (tensor<1x!tf_type.resource>) -> tensor<i64>
  // CHECK-NEXT:  return [[SIZE]] : tensor<i64>
  func.return %1 : tensor<i64>
}

// -----

// Test for case with import and find ops.
func.func @hashtable_import_then_find(%arg0: tensor<5x!tf_type.string>) -> tensor<*xi64> {
  %cst = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  %cst_0 = arith.constant dense<-1> : tensor<i64>
  %cst_1 = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_1) {device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
  %1 = "tf.LookupTableFindV2"(%0, %arg0, %cst_0) {device = ""} : (tensor<!tf_type.resource>, tensor<5x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-LABEL: hashtable_import_then_find
  // CHECK:       [[CST:%.*]] = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = arith.constant dense<-1> : tensor<i64>
  // CHECK-NEXT:  [[CST_1:%.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf_type.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_1]]) : (tensor<1x!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], %arg0, [[CST_0]]) : (tensor<1x!tf_type.resource>, tensor<5x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-NEXT:  return [[FIND]] : tensor<*xi64>
  func.return %1 : tensor<*xi64>
}

// -----

// Test for case with import and size ops.
func.func @hashtable_import_then_size(%arg0: tensor<5x!tf_type.string>) -> tensor<i64> {
  %cst = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
  // CHECK-LABEL: hashtable_import_then_size
  // CHECK:       [[CST:%.*]] = arith.constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf_type.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf_type.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf_type.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_0]]) : (tensor<1x!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
  // CHECK-NEXT:  [[SIZE:%.*]] = "tfl.hashtable_size"([[HASH_TABLE]]) : (tensor<1x!tf_type.resource>) -> tensor<i64>
  // CHECK-NEXT:  return [[SIZE]] : tensor<i64>
  func.return %1 : tensor<i64>
}

// -----

// Test for case with unsupported LookupTableRemoveV2 op.
func.func @no_legalization_on_hashtable_remove(%arg0: tensor<i64>) {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf_type.resource>, tensor<i64>) -> ()
  // CHECK-LABEL: no_legalization_on_hashtable_remove
  // CHECK-NEXT:  "tf.HashTableV2"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  func.return
}

// -----

// Test for case with unsupported MutableHashTableV2 op.
func.func @no_legalization_on_mutable_hashtable(%arg0: tensor<i64>) {
  %0 = "tf.MutableHashTableV2Op"() {container = "", key_dtype = i64, use_node_name_sharing = false, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf_type.resource>, tensor<i64>) -> ()
  // CHECK-LABEL: no_legalization_on_mutable_hashtable
  // CHECK-NEXT:  "tf.MutableHashTableV2Op"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  func.return
}
