// RUN: tf-opt -verify-diagnostics -tf-saved-model-lift-variables-test -split-input-file %s | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  // Test case: Freezing VarHandleOp ops.

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf_type.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf_type.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    func.return %4 : tensor<100x50xf32>
  }
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/kernel"
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/bias"
  // CHECK:  func @serving_default(
  // CHECK:    %arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf_saved_model.bound_input = @"dense/kernel"},
  // CHECK:    %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf_saved_model.bound_input = @"dense/bias"})
}

// -----

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  // Test case: Freezing shared VarHandleOp ops.

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf_type.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf_type.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    func.return %4 : tensor<100x50xf32>
  }

  func.func @f2(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["f2"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "", shared_name = "dense/kernel"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "", shared_name = "dense/bias"} : () -> tensor<!tf_type.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf_type.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    func.return %4 : tensor<100x50xf32>
  }
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/kernel"
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/bias"
  // CHECK:  func @f(
  // CHECK:    %arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf_saved_model.bound_input = @"dense/kernel"},
  // CHECK:    %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf_saved_model.bound_input = @"dense/bias"})

  // CHECK:  func @f2(
  // CHECK:    %arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf_saved_model.bound_input = @"dense/kernel"},
  // CHECK:    %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf_saved_model.bound_input = @"dense/bias"})
}

// -----

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  // Test case: Fix bound_inputs' types.

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<*xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf_type.resource<tensor<*xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<*xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
    %2 = "tf.Add"(%0, %1) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/kernel"
  // CHECK: "tf_saved_model.global_tensor"()
  // CHECK:    sym_name = "dense/bias"
  // CHECK:  func @serving_default(
  // CHECK:    %arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf_saved_model.bound_input = @"dense/kernel"},
  // CHECK:    %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf_saved_model.bound_input = @"dense/bias"})
}
