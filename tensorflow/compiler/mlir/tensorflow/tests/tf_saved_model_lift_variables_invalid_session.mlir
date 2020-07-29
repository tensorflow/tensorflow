// RUN: tf-opt -verify-diagnostics -tf-saved-model-lift-variables-invalid-session-test -split-input-file %s | FileCheck %s --dump-input=fail

// Test case: Invalid session.
// expected-error @+1 {{'module' op no session provided}}
module attributes {tf_saved_model.semantics} {

  func @serving_default(%arg0: tensor<!tf.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf.resource<tensor<50xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    return %4 : tensor<100x50xf32>
  }
}

// -----

// Test case: No errors on no resource arguments.
module attributes {tf_saved_model.semantics} {

	// CHECK-LABEL: @serving_default
  func @serving_default() -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    return %4 : tensor<100x50xf32>
  }
}
