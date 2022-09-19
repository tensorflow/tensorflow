// RUN: tf-quant-opt %s -quant-merge-initializer-function-ops-to-main -allow-unregistered-dialect -mlir-disable-threading -split-input-file | FileCheck %s

// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<["test"]> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_0, %ctl_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
      %out_1, %ctl_2 = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
      %ctl_3 = tf_executor.island wraps "tf.LookupTableImportV2"(%out_1, %out, %out_0) {device = ""} : (tensor<!tf_type.resource>, tensor<1x!tf_type.string>, tensor<1xi64>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }
// The session initializer function is removed.
// CHECK-NOT: @NoOp()

  func.func private @serving_default(%arg0: tensor<?x!tf_type.string>) -> tensor<*xi64> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
    %0 = tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<-1> : tensor<i64>} : () -> tensor<i64>
      %out_0, %ctl_1 = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
      %out_1, %ctl_2 = tf_executor.island wraps "tf.LookupTableFindV2"(%out_0, %arg0, %out) {device = ""} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
      tf_executor.fetch %out_1 : tensor<*xi64>
    }
    return %0 : tensor<*xi64>
  }
// Sanity check: The contents of @serving_default is untouched.
// CHECK: func.func private @serving_default(%[[ARG_0:.*]]: tensor<?x!tf_type.string>) -> tensor<*xi64>
// CHECK-NEXT: %[[RES:.*]] = tf_executor.graph
// CHECK: %[[OUT:.*]], %[[CTL:.*]] = tf_executor.island wraps "tf.Const"()
// CHECK-NEXT: %[[OUT_0:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.HashTableV2"()
// CHECK-NEXT: %[[OUT_1:.*]], %[[CTL_2:.*]] = tf_executor.island wraps "tf.LookupTableFindV2"(%[[OUT_0]], %[[ARG_0]], %[[OUT]])
// CHECK-NEXT: tf_executor.fetch %[[OUT_1]] : tensor<*xi64>
// CHECK: return %[[RES]] : tensor<*xi64>

  func.func @main(%arg0: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["serving_default_input_vocabs:0"]}) -> (tensor<*xi64> {tf_saved_model.index_path = ["StatefulPartitionedCall:0"]})
      attributes {tf.entry_function = {inputs = "serving_default_input_vocabs:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @serving_default} : (tensor<?x!tf_type.string>) -> tensor<*xi64>
      tf_executor.fetch %out : tensor<*xi64>
    }
    return %0 : tensor<*xi64>
  }
// Sanity check: The main function's signature & attributes have not changed.
// CHECK: func.func @main(%[[ARG:.*]]: tensor<?x!tf_type.string>
// CHECK-SAME: tf_saved_model.index_path = ["serving_default_input_vocabs:0"]
// CHECK-SAME: -> (tensor<*xi64> {tf_saved_model.index_path = ["StatefulPartitionedCall:0"]})
// CHECK-SAME: tf.entry_function = {inputs = "serving_default_input_vocabs:0", outputs = "StatefulPartitionedCall:0"}
// CHECK-SAME: tf_saved_model.exported_names = ["main"]

// CHECK: %[[GRAPH_OUT:.*]] = tf_executor.graph
// CHECK-NEXT: %[[OUT:.*]], %[[CTL:.*]] = tf_executor.island wraps "tf.PartitionedCall"(%[[ARG]])
// CHECK-SAME: f = @serving_default
// Checks that the contents of @NoOp are copied here.
// CHECK-NEXT: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.Const"()
// CHECK-SAME: value = dense<"test">
// CHECK-NEXT: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.Const"()
// CHECK-SAME: value = dense<1>
// CHECK-NEXT: %[[OUT_2:.*]], %[[CTL_2:.*]] = tf_executor.island wraps "tf.HashTableV2"()
// CHECK-NEXT: %[[CTL_3:.*]] = tf_executor.island wraps "tf.LookupTableImportV2"(%[[OUT_2]], %[[OUT_0]], %[[OUT_1]])
// Checks that the control output for the initializer function is fetched.
// CHECK-NEXT: tf_executor.fetch %[[OUT]], %[[CTL_3]] : tensor<*xi64>, !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GRAPH_OUT]] : tensor<*xi64>
}

// -----

// Test the case where the initializer function accepts an argument but it
// is not used within the body.
// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  "tf_saved_model.asset"() {filename = "assets/file.txt", sym_name = "__tf_saved_model_asset0_file.txt"} : () -> ()

  func.func @NoOp(%arg: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_file.txt}) attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<["test"]> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_0, %ctl_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
      %out_1, %ctl_2 = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
      %ctl_3 = tf_executor.island wraps "tf.LookupTableImportV2"(%out_1, %out, %out_0) {device = ""} : (tensor<!tf_type.resource>, tensor<1x!tf_type.string>, tensor<1xi64>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }
// The session initializer function is removed.
// CHECK-NOT: @NoOp()

  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// Sanity check: The main function's signature & attributes have not changed.
// CHECK: func.func @main()
// CHECK-SAME: tf_saved_model.exported_names = ["main"]

// CHECK: tf_executor.graph
// Checks that the contents of @NoOp are copied here.
// CHECK-NEXT: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.Const"()
// CHECK-SAME: value = dense<"test">
// CHECK-NEXT: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.Const"()
// CHECK-SAME: value = dense<1>
// CHECK-NEXT: %[[OUT_2:.*]], %[[CTL_2:.*]] = tf_executor.island wraps "tf.HashTableV2"()
// CHECK-NEXT: %[[CTL_3:.*]] = tf_executor.island wraps "tf.LookupTableImportV2"(%[[OUT_2]], %[[OUT_0]], %[[OUT_1]])
// Checks that the control output for the initializer function is fetched.
// CHECK-NEXT: tf_executor.fetch %[[CTL_3]] : !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return
}

// -----

// Tests no change when there's no initializer functions.
// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    return
  }
// CHECK: func.func @main()
// CHECK-NEXT: return
}

// -----

// Tests no change when there's no "tf_saved_model.session_initializer".
// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    return
  }
// CHECK: func.func @main()
// CHECK-NEXT: return
}

// -----

// Tests when the initializer function is empty.
// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    return
  }
// Check that the initializer function is removed.
// CHECK-NOT: @NoOp

  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    return
  }
// CHECK: func.func @main()
// CHECK-NEXT: return
}
