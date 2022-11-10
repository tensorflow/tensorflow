// RUN: tf-opt %s -split-input-file -tf-xla-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "tf.XlaLaunch"(%arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @pcall_func, operand_segment_sizes = array<i32: 0, 1, 0>} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "tf.XlaLaunch"(%arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @stateful_pcall_func, operand_segment_sizes = array<i32: 0, 1, 0>} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<!tf_type.resource>, %arg1: tensor<i64>) -> tensor<i64> {
  // CHECK: %0 = "tf.XlaLaunch"(%arg1, %arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @stateful_pcall_func_with_resources_in_order, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i64>, tensor<!tf_type.resource>) -> tensor<i64>
  %0 = "tf.StatefulPartitionedCall"(%arg1, %arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @stateful_pcall_func_with_resources_in_order} : (tensor<i64>, tensor<!tf_type.resource>) -> (tensor<i64>)
  func.return %0 : tensor<i64>
}

func.func @stateful_pcall_func_with_resources_in_order(%arg0 : tensor<i64>, %arg1 : tensor<!tf_type.resource>) -> tensor<i64> {
  func.return %arg0 : tensor<i64>
}

// -----

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<!tf_type.resource>, %arg1: tensor<i64>) -> tensor<i64> {
  // CHECK: "tf.XlaLaunch"(%arg1, %arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @stateful_pcall_func_with_resources, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i64>, tensor<!tf_type.resource>) -> tensor<i64>
  %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @stateful_pcall_func_with_resources} : (tensor<!tf_type.resource>, tensor<i64>) -> (tensor<i64>)
  // CHECK: "tf.XlaLaunch"(%arg1, %arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @stateful_pcall_func_with_resources, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i64>, tensor<!tf_type.resource>) -> tensor<i64>
  %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @stateful_pcall_func_with_resources} : (tensor<!tf_type.resource>, tensor<i64>) -> (tensor<i64>)
  func.return %0 : tensor<i64>
}

// CHECK-LABEL: func.func @stateful_pcall_func_with_resources
// CHECK-SAME:  (%arg0: tensor<i64>, %arg1: tensor<!tf_type.resource>) -> tensor<i64>
// CHECK:         return %arg0 : tensor<i64>
func.func @stateful_pcall_func_with_resources(%arg0 : tensor<!tf_type.resource>, %arg1: tensor<i64>) -> tensor<i64> {
  func.return %arg1 : tensor<i64>
}

// -----

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.While"(%arg0) {cond = @while_cond_func, body = @while_body_func, is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
    func.return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @while_cond_func
func.func @while_cond_func(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func.func @while_body_func
func.func @while_body_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @outer_stateful_pcall_func} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @outer_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @outer_stateful_pcall_func
func.func @outer_stateful_pcall_func(%arg0: tensor<i32>) -> (tensor<i32>) {
// CHECK: "tf.XlaLaunch"(%arg0) {_xla_compile_device_type = "CPU", device = "/device:CPU:0", function = @inner_stateful_pcall_func, operand_segment_sizes = array<i32: 0, 1, 0>} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @inner_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @inner_stateful_pcall_func
func.func @inner_stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
// CHECK: "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
