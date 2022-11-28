// RUN: tf-opt %s -split-input-file -verify-diagnostics  -tf-xla-cluster-formation | FileCheck %s

// Check that we outline the partitioned call to a device cluster (since it has
// `_xla_compile_device_type`).
// CHECK-LABEL: func.func @xla_must_compile_true
// CHECK: tf_device.cluster
// CHECK-NEXT: tf.StatefulPartitionedCall
// CHECK-NEXT: tf_device.return
// CHECK: tf.Const
// CHECK: tf.Add
func.func @xla_must_compile_true(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", _xla_compile_device_type = "CPU", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
// -----

// Check that we don't outline the partitioned call to a device cluster (since
// it does not has `_xla_compile_device_type`).
// CHECK-LABEL: func.func @xla_must_compile_false
// CHECK-NOT: tf_device.cluster
func.func @xla_must_compile_false(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @nested_calls
func.func @nested_calls(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
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
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @outer_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @outer_stateful_pcall_func
func.func @outer_stateful_pcall_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: tf_device.cluster
  // CHECK-NEXT: tf.StatefulPartitionedCall
  // CHECK-NEXT: tf_device.return
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @inner_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @inner_stateful_pcall_func
func.func @inner_stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
