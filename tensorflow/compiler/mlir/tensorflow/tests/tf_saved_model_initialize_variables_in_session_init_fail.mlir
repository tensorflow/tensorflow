// RUN: tf-opt -tf-saved-model-initialize-variables-in-session-init -verify-diagnostics -fail-to-fetch-local-device-manager %s

// expected-error@below{{No Local Device Manager}}
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }
}
