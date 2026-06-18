// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-xla-validate-inputs

// expected-error @+1 {{expects no nested calls of entry functions as they prevent graph traversal in some passes from working correctly}}
func.func @nested_entry_functions() attributes {tf.entry_function = {}} {
  tf_executor.graph {
     %control = tf_executor.island wraps "tf.StatefulPartitionedCall"() {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : () -> ()
     tf_executor.fetch
  }
  func.return
}

func.func @func() attributes {tf.entry_function = {}} {
  func.return
}

// -----

// expected-error @+1 {{does not support top-level compilation marker}}
func.func @top_level_compilation_marker() attributes {_xla_compile_device_type = "CPU", tf.entry_function = {}} {
  func.return
}
