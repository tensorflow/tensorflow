// RUN: tf-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -quant-insert-save-op | FileCheck %s

// SaveV2 op created for a single VarHandleOp.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
// SessionInitializerOp is untouched.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: {{.*initializers = \[@init_func_restore_op\].*}}

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"],
      tf_saved_model.initializer_type = "restore_op"} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }
// Initializer function is untouched.
// CHECK: func.func @init_func_restore_op
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
// CHECK-DAG: %[[CST:.*]] = "tf.Const"
// CHECK-DAG: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"
// CHECK: "tf.AssignVariableOp"(%[[VAR_HANDLE]], %[[CST]])

// Test that a new save function that wraps the SaveV2 op is created.
// CHECK: func.func private @tf_quant__save(%[[ARG:.*]]: tensor<!tf_type.string>)
// CHECK: %[[VAR_HANDLE:.*]] = "tf.VarHandleOp"()
// CHECK-SAME: {{.*shared_name = "var_0".*}}
// CHECK: %[[READ_VARIABLE:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE]]) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() <{{{.*value = dense<"var_0"> : tensor<1x!tf_type.string>.*}}}>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() <{{{.*value = dense<""> : tensor<1x!tf_type.string>.*}}}>
// CHECK: "tf.SaveV2"(%[[ARG]], %[[CONST_0]], %[[CONST_1]], %[[READ_VARIABLE]])
// CHECK: return
}

// -----

// SaveV2 op created for multiple VarHandleOps.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
// SessionInitializerOp is untouched.
// CHECK: "tf_saved_model.session_initializer"()
// CHECK-SAME: {{.*initializers = \[@init_func_restore_op\].*}}

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"],
      tf_saved_model.initializer_type = "restore_op"} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    %cst_1 = "tf.Const"() {value = dense<2.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
    %var_1 = "tf.VarHandleOp"() {shared_name = "var_1"} : () -> tensor<!tf_type.resource<tensor<3xf32>>>
    "tf.AssignVariableOp"(%var_1, %cst_1) : (tensor<!tf_type.resource<tensor<3xf32>>>, tensor<3xf32>) -> ()
    return
  }
// Initializer function is untouched.
// CHECK: func.func @init_func_restore_op
// CHECK-SAME: tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]
// CHECK-SAME: tf_saved_model.initializer_type = "restore_op"

// Test that a new save function that wraps the SaveV2 op is created.
// CHECK: func.func private @tf_quant__save(%[[ARG:.*]]: tensor<!tf_type.string>)
// CHECK: %[[VAR_HANDLE_0:.*]] = "tf.VarHandleOp"()
// CHECK-SAME: {{.*shared_name = "var_0".*}}
// CHECK: %[[VAR_HANDLE_1:.*]] = "tf.VarHandleOp"()
// CHECK-SAME: {{.*shared_name = "var_1".*}}

// ReadVariableOps are inserted for each VarHandleOp to read the tensor values.
// CHECK-DAG: %[[READ_VARIABLE_0:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_0]]) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
// CHECK-DAG: %[[READ_VARIABLE_1:.*]] = "tf.ReadVariableOp"(%[[VAR_HANDLE_1]]) : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>

// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() <{{{.*value = dense<\["var_0", "var_1"\]> : tensor<2x!tf_type.string>.*}}}>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() <{{{.*value = dense<""> : tensor<2x!tf_type.string>.*}}}>
// CHECK: "tf.SaveV2"(%[[ARG]], %[[CONST_0]], %[[CONST_1]], %[[READ_VARIABLE_0]], %[[READ_VARIABLE_1]])
// CHECK: return
}

// -----


// SaveV2 op not created when SessionInitializerOp doesn't exist.

module attributes {tf_saved_model.semantics} {
// CHECK-NOT: @tf_quant__save
}

// -----

// SaveV2 op not created when there are no VarHandleOp in the session
// initializer function.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"],
      tf_saved_model.initializer_type = "restore_op"} {
    return
  }
// Test that the function for SaveV2 op is not created.
// CHECK: func.func @init_func_restore_op
// CHECK-NOT: @tf_quant__save
}

// -----

// SaveV2 op not created when the initializer function doesn't exist.

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
// Test that the function for SaveV2 op is not created.
// CHECK-NOT: @tf_quant__save
}
