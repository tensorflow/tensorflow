// RUN: tf-opt -tf-saved-model-initialize-variables-in-session-init -split-input-file -verify-diagnostics %s | FileCheck %s


// Test with no session init function.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }

  // CHECK-LABEL: func @SessionInitializerFunction()
  // CHECK-SAME: tf_saved_model.exported_names = ["SessionInitializerFunction"]
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = arith.constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}

// -----

// Test with an existing session init function.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  "tf_saved_model.session_initializer"() {initializers = [@Init]} : () -> ()
  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }

  func.func @Init() attributes {tf_saved_model.exported_names = ["Init"], tf_saved_model.initializer_type = "restore_op"} {
    func.return
  }

  // CHECK-LABEL: func @Init
  // CHECK-SAME: tf_saved_model.exported_names = ["Init"]
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"()
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = arith.constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}


// -----

// Test invalid tensor
// expected-error@below{{Can't find variable var5 in session}}
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var5"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }
}

// -----

// Test with empty initializers.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }

  // CHECK-LABEL: func @SessionInitializerFunction
  // CHECK-SAME: tf_saved_model.exported_names = ["SessionInitializerFunction"]
  // CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"()
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = arith.constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}

// -----

// Variable not in session. No initialization should happen.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  // CHECK-NOT: "tf.AssignVariableOp"()

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    // CHECK: "tf.VarHandleOp"
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "invalid", shared_name = "invalid_var"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }
}
