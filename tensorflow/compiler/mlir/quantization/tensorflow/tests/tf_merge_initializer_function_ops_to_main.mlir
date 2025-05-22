// RUN: tf-quant-opt %s -tf-quant-merge-initializer-function-ops-to-main \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s
// RUN: tf-quant-opt %s -tf-quant-merge-initializer-function-ops-to-main \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -mlir-print-local-scope -mlir-print-debuginfo \
// RUN:     -verify-diagnostics | FileCheck %s --check-prefix CHECK-LOC

// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
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
// Checks that the NoOp with control dependency to the control output for the
// initializer function is created & fetched.
// CHECK-NEXT: %[[CTL_4:.*]] = tf_executor.island(%[[CTL_3]]) wraps "tf.NoOp"()
// CHECK-NEXT: tf_executor.fetch %[[OUT]], %[[CTL_4]] : tensor<*xi64>, !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GRAPH_OUT]] : tensor<*xi64>

// Checks that the location for the init op is properly set.
// CHECK-LOC-LABEL: func.func @main
// CHECK-LOC: tf_executor.island({{.*}}) wraps "tf.NoOp"()
// CHECK-LOC-SAME: loc("init_op_NoOp")
}

// -----

// Tests when the initializer function contains multiple stateful
// initialization ops. They should be transitively connected through
// control dependencies (!tf_executor.control), which is guaranteed by
// the `tf-executor-break-up-islands` pass.

// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers list is empty.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<["test_1"]> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_0, %ctl_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
      %out_1, %ctl_2 = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "1", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
      %ctl_3 = tf_executor.island wraps "tf.LookupTableImportV2"(%out_1, %out, %out_0) {device = ""} : (tensor<!tf_type.resource>, tensor<1x!tf_type.string>, tensor<1xi64>) -> ()

      %out_2, %ctl_4 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<["test_2"]> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_3, %ctl_5 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
      // Has a control dependency to the previous LookupTableImportV2.
      %out_4, %ctl_6 = tf_executor.island(%ctl_3) wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "2", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
      %ctl_7 = tf_executor.island wraps "tf.LookupTableImportV2"(%out_4, %out_2, %out_3) {device = ""} : (tensor<!tf_type.resource>, tensor<1x!tf_type.string>, tensor<1xi64>) -> ()
      tf_executor.fetch %ctl_7 : !tf_executor.control
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
// CHECK-DAG: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<"test_1">.*}}}>
// CHECK-DAG: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<1>.*}}}>

// CHECK-NEXT: %[[OUT_2:.*]], %[[CTL_2:.*]] = tf_executor.island wraps "tf.HashTableV2"()
// CHECK-NEXT: %[[CTL_3:.*]] = tf_executor.island wraps "tf.LookupTableImportV2"(%[[OUT_2]], %[[OUT_0]], %[[OUT_1]])

// CHECK-DAG: %[[OUT_3:.*]], %[[CTL_4:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<"test_2">.*}}}>
// CHECK-DAG: %[[OUT_4:.*]], %[[CTL_5:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<2>.*}}}>

// CHECK-NEXT: %[[OUT_5:.*]], %[[CTL_6:.*]] = tf_executor.island(%[[CTL_3]]) wraps "tf.HashTableV2"()
// CHECK-NEXT: %[[CTL_7:.*]] = tf_executor.island wraps "tf.LookupTableImportV2"(%[[OUT_5]], %[[OUT_3]], %[[OUT_4]])

// Checks that the NoOp with control dependency to the control output for the
// initializer function is created & fetched.
// CHECK-NEXT: %[[CTL_8:.*]] = tf_executor.island(%[[CTL_7]]) wraps "tf.NoOp"()
// CHECK-NEXT: tf_executor.fetch %[[OUT]], %[[CTL_8]] : tensor<*xi64>, !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GRAPH_OUT]] : tensor<*xi64>

// Checks that the location for the init op is properly set.
// CHECK-LOC-LABEL: func.func @main
// CHECK-LOC: tf_executor.island({{.*}}) wraps "tf.NoOp"()
// CHECK-LOC-SAME: loc("init_op_NoOp")
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

  func.func @NoOp(%arg: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_file.txt})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
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
// CHECK-NEXT: %[[CTL_4:.*]] = tf_executor.island(%[[CTL_3]]) wraps "tf.NoOp"()
// CHECK-NEXT: tf_executor.fetch %[[CTL_4]] : !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return

// Checks that the location for the init op is properly set.
// CHECK-LOC-LABEL: func.func @main
// CHECK-LOC: tf_executor.island({{.*}}) wraps "tf.NoOp"()
// CHECK-LOC-SAME: loc("init_op_NoOp")
}

// -----

// Test the case where there are 2 initializer functions ("init_op" and
// "restore_op"). The init func of type "init_op" is merged first.

// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.asset"() {filename = "assets/table.txt", sym_name = "v"} : () -> ()
  "tf_saved_model.session_initializer"() {initializers = [@NoOp_0, @NoOp_1]} : () -> ()
// Check that the initializer typed "init_op" is removed from initializers list.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = []

func.func @NoOp_0(%arg0: tensor<!tf_type.string> {tf_saved_model.bound_input = @v})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp_0"], tf_saved_model.initializer_type = "init_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      tf_executor.fetch %ctl : !tf_executor.control
    }
    return
  }
// The session initializer function is removed.
// CHECK-NOT: @NoOp_0()

  func.func @NoOp_1(%arg0: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp_1"], tf_saved_model.initializer_type = "restore_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      tf_executor.fetch %ctl : !tf_executor.control
    }
    return
  }
// The session initializer function is removed.
// CHECK-NOT: @NoOp_1()

  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// Check that the args for the "restore_op" is added before the args for the "init_op".
// CHECK: func.func @main(%[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}, %[[ARG_1:.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @v})
// CHECK-SAME: tf_saved_model.exported_names = ["main"]

// CHECK: tf_executor.graph
// Checks that the contents of the initializer functions are copied here.
// CHECK-DAG: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.Identity"(%[[ARG_0]])
// CHECK-DAG: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.Identity"(%[[ARG_1]])

// Checks that 2 `NoOp`s having control dependencies to each of the initializer
// functions are created.
// CHECK-DAG: %[[CTL_2:.*]] = tf_executor.island(%[[CTL_0]]) wraps "tf.NoOp"()
// CHECK-DAG: %[[CTL_3:.*]] = tf_executor.island(%[[CTL_1]]) wraps "tf.NoOp"()

// CHECK: tf_executor.fetch
// CHECK-SAME: !tf_executor.control, !tf_executor.control
// CHECK-NEXT: }
// CHECK-NEXT: return

// Checks that the location for the init op is properly set.
// CHECK-LOC-LABEL: func.func @main

// CHECK-LOC-DAG: tf_executor.island({{.*}}) wraps "tf.NoOp"() {{.*}} loc("init_op_NoOp_0")
// CHECK-LOC-DAG: tf_executor.island({{.*}}) wraps "tf.NoOp"() {{.*}} loc("restore_op_NoOp_1")
}

// -----

// Tests that initializer function for "restore_op" is merged into @main.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
// CHECK: "tf_saved_model.session_initializer"() <{initializers = []}>

  func.func @init_func_restore_op(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "restore_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_0, %ctl_0 = tf_executor.island wraps "tf.Const"() {value = dense<"var_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_1, %ctl_1 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "var_0", device = "/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
      %out_2, %ctl_2 = tf_executor.island wraps "tf.RestoreV2"(%arg, %out_0, %out) {} : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<2xf32>
      %ctl_3 = tf_executor.island wraps "tf.AssignVariableOp"(%out_1, %out_2) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }

  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// A new argument corresponding to the "file_prefix" should be created.
// CHECK: func.func @main(%[[ARG:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]})
// CHECK-SAME: {{{.*tf.entry_function = {inputs = "restore_op_0:0", outputs = ""}.*}}}
// CHECK-NEXT: tf_executor.graph

// Checks that the ops from @init_func_restore_op are cloned.
// CHECK-DAG: %[[CONST_0:.*]], %[[CTL:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<""> : tensor<1x!tf_type\.string>.*}}}>
// CHECK-DAG: %[[CONST_1:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.Const"() <{{{.*value = dense<"var_0"> : tensor<1x!tf_type\.string>.*}}}>
// CHECK: %[[VAR_HANDLE:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK: %[[RESTORE:.*]], %[[CTL_2:.*]] = tf_executor.island wraps "tf.RestoreV2"(%[[ARG]], %[[CONST_1]], %[[CONST_0]])
// CHECK: %[[CTL_3:.*]] = tf_executor.island wraps "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[RESTORE]])
// CHECK: %[[CTL_4:.*]] = tf_executor.island(%[[CTL_3]]) wraps "tf.NoOp"()
// CHECK-NEXT: tf_executor.fetch %[[CTL_4]] : !tf_executor.control
// CHECK: return

// Checks that the Location is properly set for the NoOp.
// CHECK-LOC: tf_executor.island({{.*}}) wraps "tf.NoOp"() {{.*}} loc("restore_op_init_func_restore_op")
}

// -----

// Test that the argument of the initializer function is correctly merged
// into @main.

// CHECK-LABEL: module
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/file.txt", sym_name = "__tf_saved_model_asset0_file.txt"} : () -> ()

  func.func @NoOp(%arg: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_file.txt})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Identity"(%arg) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      tf_executor.fetch %ctl : !tf_executor.control
    }
    return
  }

  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
  // CHECK: @main(%[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_file.txt})
  // CHECK-SAME: tf.entry_function = {inputs = "init_op_0:0", outputs = ""}
  // CHECK: %{{.*}}, %[[CTL:.*]] = tf_executor.island wraps "tf.Identity"(%[[ARG_0]])
  // CHECK: tf_executor.fetch %[[CTL]]
}

// -----

// Tests that the input name for the new argument created in @main (for the
// "restore_op" initializer function) is not added when there is no
// tf.entry_function.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
// CHECK: "tf_saved_model.session_initializer"() <{initializers = []}>

  func.func @init_func_restore_op(%arg: tensor<!tf_type.string> {tf_saved_model.index_path = ["file_prefix"]})
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "restore_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_0, %ctl_0 = tf_executor.island wraps "tf.Const"() {value = dense<"var_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %out_1, %ctl_1 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "var_0", device = "/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
      %out_2, %ctl_2 = tf_executor.island wraps "tf.RestoreV2"(%arg, %out_0, %out) {} : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<2xf32>
      %ctl_3 = tf_executor.island wraps "tf.AssignVariableOp"(%out_1, %out_2) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
      tf_executor.fetch %ctl_3 : !tf_executor.control
    }
    return
  }

  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// A new argument corresponding to the "file_prefix" should be created.
// Also checks that tf.entry_function is not created.
// CHECK: func.func @main(%[[ARG:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["file_prefix"]}) attributes {tf_saved_model.exported_names = ["main"]}
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
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// CHECK: func.func @main()
// CHECK-NEXT: tf_executor.graph {
// CHECK-NEXT: tf_executor.fetch
// CHECK-NEXT: }
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

// Tests when the main function is empty.
// CHECK-LABEL: module attributes
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// Check that the initializers attribute is untouched.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@NoOp]

  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    return
  }
// The initializer function is untouched when the main function is empty.
// CHECK: func.func @NoOp

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
  "tf_saved_model.session_initializer"() {initializers = [@init_func_when_main_empty]} : () -> ()
// Check that the initializers attribute is untouched.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@init_func_when_main_empty]

  func.func @init_func_when_main_empty()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    return
  }
// The initializer function is untouched.
// CHECK: func.func @init_func_when_main_empty()

  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
// CHECK: func.func @main()
}

// -----

// @main function must exist in a valid input module for this pass.

// expected-error @+1 {{Main function op not found.}}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()

  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    return
  }
}

// -----

// Tests malformed initializer function that has a fetch other than
// tf_executor::ControlType.

// expected-error @+1 {{Validation on initializer functions failed.}}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()

  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
      // expected-error @+1 {{Validation failed for the initializer function: NoOp. All initializer function's fetches should be tf_executor::ControlType. Got: tensor<1xi64>.}}
      tf_executor.fetch %out : tensor<1xi64>
    }
    return
  }

  func.func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}, tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
}

// -----

// Tests that an error is emitted when an initializer function does not have the
// tf_saved_model.initializer_type attribute.

// expected-error @below {{Validation on initializer functions failed.}}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1228 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()

  // expected-error @below {{Initializer func op does not have tf_saved_model.initializer_type attribute. Func op: NoOp}}
  func.func @NoOp()
    attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    tf_executor.graph {
      %out, %ctl = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
      tf_executor.fetch %ctl : !tf_executor.control
    }
    return
  }

  func.func @main() attributes {tf_saved_model.exported_names = ["main"]} {
    tf_executor.graph {
      tf_executor.fetch
    }
    return
  }
}
