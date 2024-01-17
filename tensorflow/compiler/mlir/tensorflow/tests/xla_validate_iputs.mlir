// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-xla-validate-inputs

// expected-error @+1 {{TF2XLA MLIR CPU/GPU phase 1 bridge expects no nested calls of entry functions as they prevent graph traversal in some passes from working correctly}}
func.func @nested_entry_functions(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  func.return %arg0 : tensor<i32>
}

// -----

// expected-error @+1 {{TF2XLA MLIR CPU/GPU MLIR phase 1 bridge expects single region and single block in an entry function}}
func.func @multi_blocks_entry_function(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  cf.br ^bb1
^bb1:
  func.return %arg0 : tensor<i32>
}
