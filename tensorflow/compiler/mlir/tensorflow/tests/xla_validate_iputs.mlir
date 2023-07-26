// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-xla-validate-inputs

// expected-error @+1 {{CPU/GPU MLIR phase 1 pipeline does not support nested calls of entry functions}}
func.func @nested_entry_functions(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  func.return %arg0 : tensor<i32>
}
