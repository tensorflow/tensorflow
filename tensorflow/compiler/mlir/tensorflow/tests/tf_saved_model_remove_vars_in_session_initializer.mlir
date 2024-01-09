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
  func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"], tf_saved_model.initializer_type = "restore_op"} {
    cf.br ^bb1
    ^bb1:
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // Test case: No variables
  // CHECK: func @init()
  // CHECK: tf.Const
  // CHECK: return
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()
  func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"], tf_saved_model.initializer_type = "restore_op"} {
    "tf.Const"() {value = dense<[1.0]> : tensor<1xf32> } : () -> tensor<1xf32>
    func.return
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
  func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"], tf_saved_model.initializer_type = "restore_op"} {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8xi32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "w"} : () -> tensor<*x!tf_type.resource<tensor<2xi32>>>
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %2) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    "tf.AssignAddVariableOp"(%1, %3) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2xi32>>>, tensor<i32>) -> ()
    func.return
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
  func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"], tf_saved_model.initializer_type = "restore_op"} {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<2x8xi32>>>
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "w"} : () -> tensor<*x!tf_type.resource<tensor<2xi32>>>
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %4 = "tf.Add"(%2, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.AssignAddVariableOp"(%0, %4) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2x8xi32>>>, tensor<i32>) -> ()
    "tf.AssignAddVariableOp"(%1, %4) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<2xi32>>>, tensor<i32>) -> ()
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  // Test case: Variable removed for the initializer function whose
  // initializer_type is "init_op".
  "tf_saved_model.session_initializer"() { initializers = [@init] } : () -> ()

  func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer"], tf_saved_model.initializer_type = "init_op"} {
    %0 = "tf.VarHandleOp"() {shared_name = "vocab_file"} : () -> tensor<*x!tf_type.resource<tensor<!tf_type.string>>>
    %1 = "tf.Const"() {value = dense<"assets/vocab_file.txt"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    "tf.AssignVariableOp"(%0, %1) : (tensor<*x!tf_type.resource<tensor<!tf_type.string>>>, tensor<!tf_type.string>) -> ()
    %2 = "tf.HashTableV2"() {key_dtype = !tf_type.string, shared_name = "table", value_dtype = i64} : () -> tensor<!tf_type.resource>
    %3 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource<tensor<!tf_type.string>>>) -> tensor<!tf_type.string>
    "tf.InitializeTableFromTextFileV2"(%2, %3) {key_index = -2 : i64, value_index = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
    func.return
  }
  // CHECK: func @init()
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.Const
  // CHECK-NOT: tf.AssignVariableOp
  // CHECK-NOT: tf.HashTableV2
  // CHECK-NOT: tf.ReadVariableOp
  // CHECK: return
}
