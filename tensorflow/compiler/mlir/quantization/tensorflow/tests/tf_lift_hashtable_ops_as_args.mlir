// RUN: tf-quant-opt %s -split-input-file -tf-quant-lift-hashtable-ops-as-args | FileCheck %s
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1506 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()
  func.func @init_all_tables() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init_all_tables"], tf_saved_model.initializer_type = "init_op"} {
    %cst = "tf.Const"() {value = dense<["hello", "model", "quantization"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %cst_0 = "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
    return
  }

// Check that HashTable op in the initilizer is not lifted.
// CHECK: func.func @init_all_tables()
// CHECK: %[[OUT_0:.*]] = "tf.HashTableV2"()
// CHECK: "tf.LookupTableImportV2"(%[[OUT_0]]
  func.func private @serving_default(%arg0: tensor<?x!tf_type.string> ) -> (tensor<*xi64>) attributes {tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}} {
    %cst = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %cst_0 = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {value = dense<0.00235294132> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {value = dense<0.00156862743> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
    %2 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
    %3 = "tf.StringToHashBucketFast"(%arg0) {device = "", num_buckets = 5 : i64} : (tensor<?x!tf_type.string>) -> tensor<?xi64>
    %4 = "tf.AddV2"(%3, %1) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %5 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
    return %5 : tensor<*xi64>
  }

// Check that HashTable op is lifted.
// CHECK: func.func private @serving_default
// CHECK-SAME: (%arg0: tensor<?x!tf_type.string>, %arg1: tensor<!tf_type.resource>) -> tensor<*xi64>
// CHECK-SAME: tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0,hash_table_1:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}
// CHECK: "tf.LookupTableSizeV2"(%arg1)
// CHECK: "tf.LookupTableFindV2"(%arg1
  func.func @main(%arg0: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["input_vocabs:0"]} ) -> (tensor<*xi64>  {tf_saved_model.index_path = ["FakeQuantWithMinMaxArgs_2:0"]}) attributes {tf.entry_function = {inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @serving_default} : (tensor<?x!tf_type.string>) -> (tensor<*xi64>)
    %1 = "tf.Identity"(%0) : (tensor<*xi64>) -> tensor<*xi64>
    return %1 : tensor<*xi64>
  }

// Check that the caller is updated.
// CHECK: func.func @main
// CHECK: %[[OUT_1:.*]] = "tf.HashTableV2"()
// CHECK: %[[OUT_2:.*]] = "tf.PartitionedCall"(%arg0, %[[OUT_1]])
}
// -----
// Test nested function case.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1506 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()
  func.func @init_all_tables() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init_all_tables"], tf_saved_model.initializer_type = "init_op"} {
    %cst = "tf.Const"() {value = dense<["hello", "model", "quantization"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %cst_0 = "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
    return
  }

// Check that HashTable op in the initilizer is not lifted.
// CHECK: func.func @init_all_tables()
// CHECK: %[[OUT_0:.*]] = "tf.HashTableV2"()
// CHECK: "tf.LookupTableImportV2"(%[[OUT_0]]
  func.func private @serving_default(%arg0: tensor<?x!tf_type.string> ) -> (tensor<*xi64>) attributes {tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}} {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @serving_default1} : (tensor<?x!tf_type.string>) -> (tensor<*xi64>)
    %1 = "tf.Identity"(%0) : (tensor<*xi64>) -> tensor<*xi64>
    return %1 : tensor<*xi64>
  }
// Check that HashTable op is passed through.
// CHECK: func.func private @serving_default
// CHECK-SAME: (%arg0: tensor<?x!tf_type.string>, %arg1: tensor<!tf_type.resource>) -> tensor<*xi64>
// CHECK-SAME: tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0,hash_table_1:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}
// CHECK: "tf.PartitionedCall"(%arg0, %arg1)
  func.func private @serving_default1(%arg0: tensor<?x!tf_type.string> ) -> (tensor<*xi64>) {
    %cst = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %cst_0 = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {value = dense<0.00235294132> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {value = dense<0.00156862743> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
    %2 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
    %3 = "tf.StringToHashBucketFast"(%arg0) {device = "", num_buckets = 5 : i64} : (tensor<?x!tf_type.string>) -> tensor<?xi64>
    %4 = "tf.AddV2"(%3, %1) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %5 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
    return %5 : tensor<*xi64>
  }

// Check that HashTable op is lifted.
// CHECK: func.func private @serving_default1
// CHECK-SAME: (%arg0: tensor<?x!tf_type.string>, %arg1: tensor<!tf_type.resource>) -> tensor<*xi64>
// CHECK: "tf.LookupTableSizeV2"(%arg1)
// CHECK: "tf.LookupTableFindV2"(%arg1
  func.func @main(%arg0: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["input_vocabs:0"]} ) -> (tensor<*xi64>  {tf_saved_model.index_path = ["FakeQuantWithMinMaxArgs_2:0"]}) attributes {tf.entry_function = {inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @serving_default} : (tensor<?x!tf_type.string>) -> (tensor<*xi64>)
    %1 = "tf.Identity"(%0) : (tensor<*xi64>) -> tensor<*xi64>
    return %1 : tensor<*xi64>
  }
// Check that the caller is updated.
// CHECK: func.func @main
// CHECK: %[[OUT_1:.*]] = "tf.HashTableV2"()
// CHECK: %[[OUT_2:.*]] = "tf.PartitionedCall"(%arg0, %[[OUT_1]])
}

// -----

// Test multiple HashTable ops.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1506 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> ()
  func.func @init_all_tables() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init_all_tables"], tf_saved_model.initializer_type = "init_op"} {
    %cst = "tf.Const"() {value = dense<["hello", "model", "quantization"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %cst_0 = "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_0", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%0, %cst, %cst_0) {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
    %1 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%1, %cst, %cst_0) {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
    return
  }
// Check that HashTable op in the initilizer is not lifted.
// CHECK: func.func @init_all_tables()
// CHECK: %[[OUT_0:.*]] = "tf.HashTableV2"()
// CHECK: "tf.LookupTableImportV2"(%[[OUT_0]]

  func.func private @serving_default(%arg0: tensor<?x!tf_type.string> ) -> (tensor<*xi64>) attributes {tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}} {
    %cst = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %cst_0 = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {value = dense<0.00235294132> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {value = dense<0.00156862743> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %1 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_0", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %2 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
    %3 = "tf.LookupTableSizeV2"(%1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
    %4 = "tf.AddV2"(%2, %3) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %5 = "tf.LookupTableFindV2"(%0, %arg0, %cst) {device = ""} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
    %6 = "tf.AddV2"(%5, %4) {device = ""} : (tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
    return %6 : tensor<*xi64>
  }
// Check that HashTable op is lifted.
// CHECK: func.func private @serving_default
// CHECK-SAME: (%arg0: tensor<?x!tf_type.string>, %arg1: tensor<!tf_type.resource>, %arg2: tensor<!tf_type.resource>) -> tensor<*xi64>
// CHECK-SAME: tf.entry_function = {control_outputs = "", inputs = "input_vocabs:0,hash_table_1:0,hash_table_2:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}
// CHECK: "tf.LookupTableSizeV2"(%arg1)
// CHECK: "tf.LookupTableSizeV2"(%arg2)
// CHECK: "tf.LookupTableFindV2"(%arg1

  func.func @main(%arg0: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["input_vocabs:0"]} ) -> (tensor<*xi64>  {tf_saved_model.index_path = ["FakeQuantWithMinMaxArgs_2:0"]}) attributes {tf.entry_function = {inputs = "input_vocabs:0", outputs = "FakeQuantWithMinMaxArgs_2:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @serving_default} : (tensor<?x!tf_type.string>) -> (tensor<*xi64>)
    %1 = "tf.Identity"(%0) : (tensor<*xi64>) -> tensor<*xi64>
    return %1 : tensor<*xi64>
  }

// Check that the caller is updated.
// CHECK: func.func @main
// CHECK: %[[HASHTABLE_1:.*]] = "tf.HashTableV2"()
// CHECK: %[[HASHTABLE_2:.*]] = "tf.HashTableV2"()
// CHECK: %[[OUT_2:.*]] = "tf.PartitionedCall"(%arg0, %[[HASHTABLE_1]], %[[HASHTABLE_2]])
}
