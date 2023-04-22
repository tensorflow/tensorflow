// RUN: tf-opt %s -split-input-file -tfl-legalize-hashtables-tf --cse | FileCheck %s

// Test for case with string -> int64 hashtable.
func @hashtable_string_to_int64(%arg0: tensor<i64>) -> tensor<*xi64> {
  %cst = constant dense<"f"> : tensor<!tf.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  %1 = "tf.LookupTableFindV2"(%0, %cst, %arg0) {device = ""} : (tensor<!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-LABEL: hashtable_string_to_int64
  // CHECK:       [[CST:%.*]] = constant dense<"f"> : tensor<!tf.string>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf.string, table_id = 1530976467 : i32, value_dtype = i64} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], [[CST]], %arg0) : (tensor<1x!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-NEXT:  return [[FIND]] : tensor<*xi64>
  return %1 : tensor<*xi64>
}

// -----

// Test for case with int64 -> string hashtable.
func @hashtable_int64_to_string(%arg0: tensor<i64>) -> tensor<*x!tf.string> {
  %cst = constant dense<"f"> : tensor<!tf.string>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
  %1 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf.resource>, tensor<i64>, tensor<!tf.string>) -> tensor<*x!tf.string>
  // CHECK-LABEL: hashtable_int64_to_string
  // CHECK:       [[CST:%.*]] = constant dense<"f"> : tensor<!tf.string>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = i64, table_id = 1530976467 : i32, value_dtype = !tf.string} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], %arg0, [[CST]]) : (tensor<1x!tf.resource>, tensor<i64>, tensor<!tf.string>) -> tensor<*x!tf.string>
  // CHECK-NEXT:  return [[FIND]] : tensor<*x!tf.string>
  return %1 : tensor<*x!tf.string>
}

// -----

// Test for case with unsupported string -> string mapping.
func @no_legalization_on_hashtable_string_to_string(%arg0: tensor<!tf.string>) {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
  // CHECK-LABEL: no_legalization_on_hashtable_string_to_string
  // CHECK-NEXT:  "tf.HashTableV2"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  return
}

// -----

// Test for case with import op.
func @hashtable_import(%arg0: tensor<5x!tf.string>) {
  %cst = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  %cst_0 = constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
  return
  // CHECK-LABEL: hashtable_import
  // CHECK:       [[CST:%.*]] = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_0]]) : (tensor<1x!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
}

// -----

// Test for case with size op.
func @hashtable_size(%arg0: tensor<5x!tf.string>) -> tensor<i64> {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf.resource>) -> tensor<i64>
  // CHECK-LABEL: hashtable_size
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:  [[SIZE:%.*]] = "tfl.hashtable_size"([[HASH_TABLE]]) : (tensor<1x!tf.resource>) -> tensor<i64>
  // CHECK-NEXT:  return [[SIZE]] : tensor<i64>
  return %1 : tensor<i64>
}

// -----

// Test for case with import and find ops.
func @hashtable_import_then_find(%arg0: tensor<5x!tf.string>) -> tensor<*xi64> {
  %cst = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  %cst_0 = constant dense<-1> : tensor<i64>
  %cst_1 = constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_1) {device = ""} : (tensor<!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
  %1 = "tf.LookupTableFindV2"(%0, %arg0, %cst_0) {device = ""} : (tensor<!tf.resource>, tensor<5x!tf.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-LABEL: hashtable_import_then_find
  // CHECK:       [[CST:%.*]] = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = constant dense<-1> : tensor<i64>
  // CHECK-NEXT:  [[CST_1:%.*]] = constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_1]]) : (tensor<1x!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
  // CHECK-NEXT:  [[FIND:%.*]] = "tfl.hashtable_find"([[HASH_TABLE]], %arg0, [[CST_0]]) : (tensor<1x!tf.resource>, tensor<5x!tf.string>, tensor<i64>) -> tensor<*xi64>
  // CHECK-NEXT:  return [[FIND]] : tensor<*xi64>
  return %1 : tensor<*xi64>
}

// -----

// Test for case with import and size ops.
func @hashtable_import_then_size(%arg0: tensor<5x!tf.string>) -> tensor<i64> {
  %cst = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  %cst_0 = constant dense<[0, 1, 2]> : tensor<3xi64>
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_1dd4fef4-646d-491f-a3a8-bf5334f45813", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
  "tf.LookupTableImportV2"(%0, %cst, %cst_0) {device = ""} : (tensor<!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf.resource>) -> tensor<i64>
  // CHECK-LABEL: hashtable_import_then_size
  // CHECK:       [[CST:%.*]] = constant dense<["emerson", "lake", "palmer"]> : tensor<3x!tf.string>
  // CHECK-NEXT:  [[CST_0:%.*]] = constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT:  [[HASH_TABLE:%.*]] = "tfl.hashtable"() {key_dtype = !tf.string, table_id = -1323619995 : i32, value_dtype = i64} : () -> tensor<1x!tf.resource>
  // CHECK-NEXT:   "tfl.hashtable_import"([[HASH_TABLE]], [[CST]], [[CST_0]]) : (tensor<1x!tf.resource>, tensor<3x!tf.string>, tensor<3xi64>) -> ()
  // CHECK-NEXT:  [[SIZE:%.*]] = "tfl.hashtable_size"([[HASH_TABLE]]) : (tensor<1x!tf.resource>) -> tensor<i64>
  // CHECK-NEXT:  return [[SIZE]] : tensor<i64>
  return %1 : tensor<i64>
}

// -----

// Test for case with unsupported LookupTableRemoveV2 op.
func @no_legalization_on_hashtable_remove(%arg0: tensor<i64>) {
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = i64, shared_name = "hash_table_e308c10b-91c8-416c-81f9-af5bf6aba847", use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf.resource>, tensor<i64>) -> ()
  // CHECK-LABEL: no_legalization_on_hashtable_remove
  // CHECK-NEXT:  "tf.HashTableV2"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  return
}

// -----

// Test for case with unsupported MutableHashTableV2 op.
func @no_legalization_on_mutable_hashtable(%arg0: tensor<i64>) {
  %0 = "tf.MutableHashTableV2Op"() {container = "", key_dtype = i64, use_node_name_sharing = false, value_dtype = !tf.string} : () -> tensor<!tf.resource>
  "tf.LookupTableRemoveV2"(%0, %arg0) {device = ""} : (tensor<!tf.resource>, tensor<i64>) -> ()
  // CHECK-LABEL: no_legalization_on_mutable_hashtable
  // CHECK-NEXT:  "tf.MutableHashTableV2Op"
  // CHECK-NEXT:  "tf.LookupTableRemoveV2"
  return
}
