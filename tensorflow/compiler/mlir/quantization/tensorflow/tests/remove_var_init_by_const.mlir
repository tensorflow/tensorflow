// RUN: tf-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -quant-remove-var-init-by-const | FileCheck %s

// Single `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` pattern removed from
// the initializer function.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"()
  // CHECK-SAME: initializers = [@init_func_restore_op]

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }
  // All three ops should have been removed.
  // CHECK: @init_func_restore_op
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK-NEXT: return
}

// -----

// The `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` pattern is not removed
// from the initializer function that is not "restore_op" type.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_init_op]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"()
  // CHECK-SAME: initializers = [@init_func_init_op]

  func.func @init_func_init_op() -> () attributes {
      tf_saved_model.initializer_type = "init_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_init_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }
  // Nothing has been removed.
  // CHECK: @init_func_init_op
  // CHECK-NEXT: "tf.Const"
  // CHECK-NEXT: "tf.VarHandleOp"
  // CHECK-NEXT: "tf.AssignVariableOp"
  // CHECK-NEXT: return
}

// -----

// If `tf.Const` is not used to initialize the variable, it is not removed.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"()
  // CHECK-SAME: initializers = [@init_func_restore_op]

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %var_0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    %add_0 = "tf.Identity"(%cst_0) : (tensor<2xf32>) -> tensor<2xf32>
    %var_1 = "tf.VarHandleOp"() {shared_name = "var_1"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf.AssignVariableOp"(%var_1, %add_0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    return
  }
  // The second AssignVariableOp, which takes the result of the `tf.Identity`
  // op, is not removed. Note that the first AssignVariableOp is removed.
  // CHECK: @init_func_restore_op
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: %[[CST:.*]] = "tf.Const"()
  // CHECK-NEXT: %[[IDENTITY:.*]] = "tf.Identity"(%[[CST]])
  // CHECK-NEXT: %[[VAR:.*]] = "tf.VarHandleOp"() {{{.*shared_name = "var_1".*}}}
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[VAR]], %[[IDENTITY]])
}

// -----

// If something other than `tf.VarHandleOp` is being initialized, it is
// not erased.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"()
  // CHECK-SAME: initializers = [@init_func_restore_op]

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    // Note: this is a contrived example and is an invalid input.
    %var_0 = "tf.HashTableV2"() {key_dtype = i64, value_dtype = !tf_type.string} : () -> tensor<!tf_type.resource>
    "tf.AssignVariableOp"(%var_0, %cst_0) : (tensor<!tf_type.resource>, tensor<2xf32>) -> ()
    return
  }
  // CHECK: @init_func_restore_op
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK: %[[CST:.*]] = "tf.Const"()
  // CHECK-NEXT: %[[HASH_TABLE:.*]] = "tf.HashTableV2"()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[HASH_TABLE]], %[[CST]])
}

// -----


// Nothing happens when there are no `tf_saved_model.session_initializer`.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
}

// -----

// Nothing happens when there are no initializer functions.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
}

// -----

// Nothing happens when the initializer function of type = "restore_op" is
// empty.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()
  // CHECK: "tf_saved_model.session_initializer"()
  // CHECK-SAME: initializers = [@init_func_restore_op]

  func.func @init_func_restore_op() -> () attributes {
      tf_saved_model.initializer_type = "restore_op",
      tf_saved_model.exported_names = ["tf_saved_model.session_initializer_restore_op"]} {
    return
  }
  // CHECK: @init_func_restore_op
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK-NEXT: return
}
