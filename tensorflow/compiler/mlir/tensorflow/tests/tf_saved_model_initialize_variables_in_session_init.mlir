// RUN: tf-opt -tf-saved-model-initialize-variables-in-session-init-test -split-input-file -verify-diagnostics %s | FileCheck %s


// Test with no session init function.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    return %1 : tensor<100x50xf32>
  }

  // CHECK-LABEL: func @SessionInitializerFunction()
  // CHECK-SAME: {tf_saved_model.exported_names = ["SessionInitializerFunction"]}
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}

// -----

// Test with an existing session init function.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  "tf_saved_model.session_initializer"() {initializers = [@Init]} : () -> ()
  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    return %1 : tensor<100x50xf32>
  }

  func @Init() attributes {tf_saved_model.exported_names = ["Init"]} {
    return
  }

  // CHECK-LABEL: func @Init
  // CHECK-SAME: {tf_saved_model.exported_names = ["Init"]}
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"()
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}


// -----

// Test invalid tensor
// expected-error@below{{Can't find variable var5 in session}}
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var5"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    return %1 : tensor<100x50xf32>
  }
}

// -----

// Test with empty initializers.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    return %1 : tensor<100x50xf32>
  }

  // CHECK-LABEL: func @SessionInitializerFunction
  // CHECK-SAME: {tf_saved_model.exported_names = ["SessionInitializerFunction"]}
  // CHECK: %[[VAR:.*]] = "tf.VarHandleOp"()
  // CHECK-SAME: "var1"
  // CHECK: %[[CST:.*]] = constant dense<> : tensor<0xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[CST]])
}

// -----

// expected-error@below{{failed to fetch variable from Session}}
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "invalid", shared_name = "invalid_var"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    return %1 : tensor<100x50xf32>
  }
}
