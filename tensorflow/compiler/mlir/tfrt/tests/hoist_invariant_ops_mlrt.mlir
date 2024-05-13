// RUN: tf-tfrt-opt -split-input-file -tfrt-lower-tf-savedmodel="hoist-invariant-ops=true fuse-get-resource-ops=false" %s | FileCheck %s --dump-input=fail --dump-input-filter=all

module attributes {tf_saved_model.semantics} {

// Test hoisting hash table op.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[handle:%.*]] = "tf.HashTableV2"()
// CHECK-SAME: shared_name = "x"
// CHECK: "tf._TfrtSetResource"([[handle]]) <{index = [[handle_id:.*]] : i64}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
// CHECK: [[x:%.*]] = "tf.LookupTableSizeV2"([[handle]])
// CHECK: "tf._TfrtSetResource"([[x]]) <{index = [[size_id:.*]] : i64}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i64>) -> ()

// CHECK: func @test_hoist_hash_table
func.func @hoist_hash_table(%arg: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["input"]}, %default: tensor<i64> {tf_saved_model.index_path = ["default"]}) -> (tensor<i64> {tf_saved_model.index_path = ["r"]}, tensor<*xi64> {tf_saved_model.index_path = ["r1"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_hash_table"]} {
  // CHECK-NOT: tf.HashTableV2
  // CHECK-NOT: tf.LookupTableSizeV2
  // CHECK-DAG: [[v0:%.*]] = "tf._TfrtGetResource"() <{container = [""], indices = [[[handle_id]]], shared_name = [{{.*}}]}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK-DAG: [[v1:%.*]] = "tf._TfrtGetResource"() <{container = [""], indices = [[[size_id]]], shared_name = [{{.*}}]}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK-DAG: [[r:%.*]] = "tf.LookupTableFindV2"([[v0]]
  // CHECK-DAG: return [[v1]], [[r]]
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "x", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
  %2 = "tf.LookupTableFindV2"(%0, %arg, %default) {device = "/CPU:0"} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  func.return %1, %2 : tensor<i64>, tensor<*xi64>
}

}

// -----
