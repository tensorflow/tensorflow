// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-saved-model-remove-vars-in-session-initializer | FileCheck %s

module attributes {tf_saved_model.semantics} {
  // Test case: No session initializer op
}

// -----

module attributes {tf_saved_model.semantics} {
  // Test case: No matching function for the given session initializer.
  // expected-error@+1 {{'tf_saved_model.session_initializer' op the initializer function does not exist}}
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
}

// -----

module attributes {tf_saved_model.semantics} {
  // Test case: Invalid multiple blocks in the initializer funcion.
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
  // expected-error@+1 {{expects exactly one block in the MLIR function}}
  func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    br ^bb1
    ^bb1:
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // Test case: No variables
  // CHECK: func @init()
  // CHECK: tf.Const
  // CHECK: return
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
  func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    return
  }
}

// -----

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  // Test case: Variable removal.
  // CHECK: func @init()
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.Const
  // CHECK-NOT: tf.AssignAddVariableOp
  // CHECK: return
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
  func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<2x8xi32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "w"} : () -> tensor<*x!tf.resource<tensor<2xi32>>>
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %2) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    "tf.AssignAddVariableOp"(%1, %3) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource<tensor<2xi32>>>, tensor<i32>) -> ()
    return
  }
}

// -----

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  // Test case: Removal of shared variables.
  // CHECK: func @init()
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.Const
  // CHECK-NOT: tf.AssignAddVariableOp
  // CHECK: return
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
  func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"]} {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<2x8xi32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "w"} : () -> tensor<*x!tf.resource<tensor<2xi32>>>
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %4 = "tf.Add"(%2, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %4) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    "tf.AssignAddVariableOp"(%1, %4) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource<tensor<2xi32>>>, tensor<i32>) -> ()
    return
  }
}
