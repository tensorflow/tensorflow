// RUN: tf-opt -tf-saved-model-mark-initialized-variables-test %s | FileCheck %s
// RUN: tf-opt -tf-saved-model-mark-initialized-variables-invalid-session-test %s | FileCheck %s --check-prefix=INVALID


module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}, %arg1: tensor<!tf_type.resource<tensor<50xf32>>> {tf.resource_name = "dense/bias"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var2"} : () -> tensor<!tf_type.resource<tensor<50xf32>>>
    %3 = "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf_type.resource<tensor<50xf32>>>) -> tensor<50xf32>
    %4 = "tf.Add"(%1, %3) {device = ""} : (tensor<100x50xf32>, tensor<50xf32>) -> tensor<100x50xf32>
    %5 = "tf.VarHandleOp"() {_class = ["loc:@dense/bias"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var3"} : () -> tensor<!tf_type.resource<tensor<50xf32>>>
    func.return %4 : tensor<100x50xf32>
  }
  // CHECK: "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "var1"
  // CHECK-SAME: _is_initialized = true
  // CHECK: "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "var2"
  // CHECK-SAME: _is_initialized = true
  // CHECK: "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "var3"
  // CHECK-SAME: _is_initialized = false

  // INVALID-NOT: _is_initialized
}
