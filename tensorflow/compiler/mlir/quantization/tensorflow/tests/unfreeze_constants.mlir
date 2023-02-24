// RUN: tf-quant-opt %s -quant-unfreeze-constants='size_threshold_in_bytes=16' \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// Tests a case with one ConstOp and a tf_saved_model.session_initializer with an empty initializers.
module attributes {tf_saved_model.semantics} {

  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
// Check that the init function is created & added to the initializers attribute.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@init_func_restore_op]

// CHECK: func.func @init_func_restore_op()
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"

// Check that variable is initialized by assigning the const value within the initializer function.
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<8xf32>}
// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}}
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[CST_0]])

  func.func @serving_default() -> (tensor<8xf32> {tf_saved_model.index_path = ["output"]})
    attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst_0 = "tf.Const"() {device = "", value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
    return %cst_0 : tensor<8xf32>
  }
// Check that the ConstOp's use is replaced by VarHandleOp -> ReadVariableOp.
// CHECK: @serving_default
// CHECK-DAG: %[[VAR_HANDLE_2:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_0:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_2]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK: return %[[READ_VAR_0]] : tensor<8xf32>
}

// -----

// Tests the case when there's no tf_saved_model.session_initializer.
module attributes {tf_saved_model.semantics} {

// Check that a new tf_saved_model.session_initializer is created, along with an initialier function.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@init_func_restore_op]

// CHECK: func.func @init_func_restore_op()
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"

// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {{{.*value = dense<1.000000e\+00> : tensor<8xf32>.*}}}
// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}}
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[CST_0]])

// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() {{{.*value = dense<2.000000e\+00> : tensor<8xf32>.*}}}
// CHECK-DAG: %[[VAR_HANDLE_1:.*]] = "tf.VarHandleOp"()  {{.*shared_name = "const_1".*}}
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_1]], %[[CST_1]])

  func.func @serving_default() -> (tensor<8xf32> {tf_saved_model.index_path = ["output"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst_0 = "tf.Const"() {device = "", value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %cst_1 = "tf.Const"() {device = "", value = dense<2.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "tf.AddV2"(%cst_0, %cst_1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
// CHECK: @serving_default
// CHECK-DAG: %[[VAR_HANDLE_2:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_0:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_2]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK-DAG: %[[VAR_HANDLE_3:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_1".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_1:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_3]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK-DAG: %[[ADD_0:.*]] = "tf.AddV2"(%[[READ_VAR_0]], %[[READ_VAR_1]])
// CHECK: return %[[ADD_0]] : tensor<8xf32>
}

// -----

// Tests the case when there's a tf_saved_model.session_initializer and an empty init function.
module attributes {tf_saved_model.semantics} {

  "tf_saved_model.session_initializer"() {initializers = [@init]} : () -> ()
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@init]

  func.func @init() attributes {tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init"], tf_saved_model.initializer_type = "restore_op"} {
    return
  }
// CHECK: func.func @init()
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"

// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<8xf32>}
// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"()
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[CST_0]])

// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<8xf32>}
// CHECK-DAG: %[[VAR_HANDLE_1:.*]] = "tf.VarHandleOp"()
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_1]], %[[CST_1]])

  func.func @serving_default(%arg0: tensor<8xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<8xf32> {tf_saved_model.index_path = ["output"]})
    attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst_0 = "tf.Const"() {device = "", value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %cst_1 = "tf.Const"() {device = "", value = dense<2.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "tf.Sub"(%cst_0, %cst_1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
// CHECK: @serving_default
// CHECK-DAG: %[[VAR_HANDLE_2:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_0:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_2]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK-DAG: %[[VAR_HANDLE_3:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_1".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_1:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_3]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK-DAG: %[[SUB_0:.*]] = "tf.Sub"(%[[READ_VAR_0]], %[[READ_VAR_1]])
// CHECK: return %[[SUB_0]] : tensor<8xf32>
}

// -----

// Tests the case when there's a tf_saved_model.session_initializer and an init function whose type is "init_op".
module attributes {tf_saved_model.semantics} {

  "tf_saved_model.session_initializer"() {initializers = [@init]} : () -> ()
// Check that @init_func_restore_op is added to the initializers list.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: initializers = [@init, @init_func_restore_op]

// Check that @init_func_restore_op is newly created with variable initializations.
// CHECK: @init_func_restore_op()
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"

// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<8xf32>}
// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"()
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[CST_0]])

  func.func @init() attributes {tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init"], tf_saved_model.initializer_type = "init_op"} {
    return
  }
// Check that @init is not removed.
// CHECK: @init()
// CHECK-SAME: tf_saved_model.initializer_type = "init_op"

  func.func @serving_default() -> (tensor<8xf32> {tf_saved_model.index_path = ["output"]})
    attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst_0 = "tf.Const"() {device = "", value = dense<3.0> : tensor<8xf32>} : () -> tensor<8xf32>
    return %cst_0 : tensor<8xf32>
  }
}

// -----

// Tests the case when there is no ConstOp.
module attributes {tf_saved_model.semantics} {

// Check that nothing happens when there's no ConstOp in the graph.
// CHECK-NOT: "tf_saved_model.session_initializer"()

  func.func @serving_default(%arg_0: tensor<5xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<5xf32> {tf_saved_model.index_path = ["output"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "inputs:0", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    return %arg_0 : tensor<5xf32>
  }
// CHECK: @serving_default(%[[ARG_0:.*]]: tensor<5xf32> {{.*}})
// CHECK-NEXT: return %[[ARG_0]] : tensor<5xf32>
}

// -----

// Tests that constants that are smaller than "size_threshold_in_bytes" are
// not converted to variables. This test uses the threshold of 16 bytes.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()

  func.func @init_func_restore_op() attributes {tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init"],
                                           tf_saved_model.initializer_type = "restore_op"} {
    return
  }

  func.func @serving_default() -> (tensor<12xf32> {tf_saved_model.index_path = ["output"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "output:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    // Should be unfrozen.
    %cst_0 = "tf.Const"() {value = dense<5.0> : tensor<8xf32>} : () -> tensor<8xf32>
    // Consts below are smaller than or equal to the threshold so they
    // should not be converted to variables.
    %cst_1 = "tf.Const"() {value = dense<5.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %cst_axis = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %0 = "tf.ConcatV2"(%cst_0, %cst_1, %cst_axis) : (tensor<8xf32>, tensor<4xf32>, tensor<i64>) -> tensor<12xf32>
    return %0 : tensor<12xf32>
  }
// CHECK: func.func @init_func_restore_op()

// Check that `tf.VarHandleOp` is only created for the constant that is larger
// than the threshold (16 bytes for this test).
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {{{.*value = dense<5.000000e\+00> : tensor<8xf32>.*}}}
// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}}
// CHECK-DAG: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[CST_0]])

// Make sure that there are no more `tf.VarHandleOp`s and `tf.AssignVariableOp`s
// in this function.
// CHECK-NOT: "tf.VarHandleOp"
// CHECK-NOT: "tf.AssignVariableOp"

// Only the large constant is replaced with the `tf.VarHandleOp ->
// tf.ReadVariableOp` pattern and others remain as `tf.Const`s.
// CHECK: @serving_default
// CHECK-DAG: %[[VAR_HANDLE_2:.*]] = "tf.VarHandleOp"() {{.*shared_name = "const_0".*}} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
// CHECK-DAG: %[[READ_VAR_0:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_2]]) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() {{{.*value = dense<5.000000e\+00> : tensor<4xf32>.*}}}
// CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {{{.*value = dense<0> : tensor<i64>.*}}}
// CHECK-DAG: %[[CONCAT:.*]] = "tf.ConcatV2"(%[[READ_VAR_0]], %[[CST_1]], %[[AXIS]])
// CHECK: return %[[CONCAT]] : tensor<12xf32>
}
