// RUN: tf-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -quant-insert-restore-op | FileCheck %s
// RUN: tf-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope \
// RUN:   -quant-insert-restore-op | FileCheck %s --check-prefix CHECK-LOC

// RestoreV2 op created for a single VarHandleOp.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }

// CHECK: func.func @init_func_restore_op
// Check that an argument ("__tf_file_prefix") is created.
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}

// Original `AssignVariableOp(VarHandleOp, Const)` pattern persists.
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {{.*value = dense<1.000000e\+00> : tensor<2xf32>.*}}
// CHECK-DAG: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"() {{.*shared_name = "var_0".*}} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[CST_0]]) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() {{.*value = dense<"var_0"> : tensor<1x!tf_type.string>.*}}
// CHECK-DAG: %[[CST_2:.*]] = "tf.Const"() {{.*value = dense<""> : tensor<1x!tf_type.string>.*}}

// Test that RestoreV2 op is created with 1 resulting value.
// CHECK: %[[RESTORE:.*]] = "tf.RestoreV2"(%[[ARG_0]], %[[CST_1]], %[[CST_2]]) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<2xf32>
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[RESTORE]]) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

// Test that the loc is properly set to it's shared_name.
// CHECK-LOC: "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK-LOC-SAME: loc("var_0")
}

// -----

// RestoreV2 op created for multiple VarHandleOps.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op_multiple_variables]} : () -> ()

  func.func @init_func_restore_op_multiple_variables() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

    %cst_1 = "tf.Const"() {value = dense<2> : tensor<4xi32>} : () -> tensor<4xi32>
    %var_1 = "tf.VarHandleOp"() {shared_name = "var_1"} : () -> tensor<!tf_type.resource<tensor<4xi32>>>
    "tf.AssignVariableOp"(%var_1, %cst_1) : (tensor<!tf_type.resource<tensor<4xi32>>>, tensor<4xi32>) -> ()
    return
  }

// CHECK: func.func @init_func_restore_op_multiple_variables
// Check that an argument ("__tf_file_prefix") is created.
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}

// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"() {{.*shared_name = "var_0".*}} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
// CHECK-DAG: %[[VAR_HANDLE_1:.*]] = "tf.VarHandleOp"() {{.*shared_name = "var_1".*}} : () -> tensor<!tf_type.resource<tensor<4xi32>>>

// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() <{{{.*value = dense<\["var_0", "var_1"\]> : tensor<2x!tf_type.string>.*}}}>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() <{{{.*value = dense<""> : tensor<2x!tf_type.string>.*}}}>

// Test that RestoreV2 op is created with 2 resulting values.
// CHECK: %[[RESTORE:.*]]:2 = "tf.RestoreV2"(%[[ARG_0]], %[[CST_0]], %[[CST_1]]) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<2xf32>, tensor<4xi32>)

// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[RESTORE]]#0) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE_1]], %[[RESTORE]]#1) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<4xi32>>>, tensor<4xi32>) -> ()

// Test that the locs are properly set to their shared_names.
// CHECK-LOC: "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK-LOC-SAME: loc("var_0")
// CHECK-LOC: "tf.VarHandleOp"() <{{{.*shared_name = "var_1".*}}}>
// CHECK-LOC-SAME: loc("var_1")
}

// -----

// RestoreV2 op not created for `AssignVariableOp(VarHandleOp, Const)` patterns
// in the initializer function of "init_op" type.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_init_op]} : () -> ()

  func.func @init_func_init_op() -> () attributes {
      tf_saved_model.initializer_type = "init_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) {validate_shape = false} : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }
// Check that no function argument is created.
// CHECK: func.func @init_func_init_op()

// CHECK-DAG: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}> : () -> tensor<!tf_type.resource<tensor<2xf32>>>
// CHECK-DAG: %[[CST:.*]] = "tf.Const"() <{{{.*value = dense<1.000000e\+00> : tensor<2xf32>.*}}}>
// Make sure that "tf.RestoreV2" is not created.
// CHECK-NOT: "tf.RestoreV2"
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[CST]]) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

// CHECK-LOC: @init_func_init_op
// CHECK-LOC: return
}

// -----

// Test that `RestoreV2Op` is created even when the `Const` op is shared across
// `AssignVariableOp`s.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op_multiple_variables_sharing_const]} : () -> ()

  func.func @init_func_restore_op_multiple_variables_sharing_const() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    // This const is shared and initializes two variables.
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>

    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

    %var_1 = "tf.VarHandleOp"() {shared_name = "var_1"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_1, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }

// CHECK: func.func @init_func_restore_op_multiple_variables_sharing_const
// Check that an argument ("__tf_file_prefix") is created.
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.string> {tf_saved_model.index_path = ["__tf_file_prefix"]}

// CHECK-DAG: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"() {{.*shared_name = "var_0".*}} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
// CHECK-DAG: %[[VAR_HANDLE_1:.*]] = "tf.VarHandleOp"() {{.*shared_name = "var_1".*}} : () -> tensor<!tf_type.resource<tensor<2xf32>>>

// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() <{{{.*value = dense<\["var_0", "var_1"\]> : tensor<2x!tf_type.string>.*}}}>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() <{{{.*value = dense<""> : tensor<2x!tf_type.string>.*}}}>

// Test that RestoreV2 op is created with 2 resulting values.
// CHECK: %[[RESTORE:.*]]:2 = "tf.RestoreV2"(%[[ARG_0]], %[[CST_0]], %[[CST_1]]) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<2xf32>, tensor<2xf32>)

// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE_0]], %[[RESTORE]]#0) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE_1]], %[[RESTORE]]#1) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

// Test that the locs are properly set to their shared_names.
// CHECK-LOC: "tf.VarHandleOp"() <{{{.*shared_name = "var_0".*}}}>
// CHECK-LOC-SAME: loc("var_0")
// CHECK-LOC: "tf.VarHandleOp"() <{{{.*shared_name = "var_1".*}}}>
// CHECK-LOC-SAME: loc("var_1")
}


// -----

// Test that "tf.RestoreV2" is not created because there are no variables.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op_no_variable]} : () -> ()

  func.func @init_func_restore_op_no_variable() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    return
  }
// CHECK: func.func @init_func_restore_op_no_variable()
// CHECK-NOT: "tf.RestoreV2"
}

// -----

// Test when there are no initializers.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
// CHECK-NOT: "tf.RestoreV2"
}

// -----

// Test when there is no SessionInitializerOp.

module attributes {tf_saved_model.semantics} {
// CHECK-NOT: "tf.RestoreV2"
}
