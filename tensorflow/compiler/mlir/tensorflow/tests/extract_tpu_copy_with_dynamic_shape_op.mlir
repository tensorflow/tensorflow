// RUN: tf-opt -split-input-file -verify-diagnostics -tf-extract-tpu-copy-with-dynamic-shape-op %s | FileCheck %s

// Test that extract TPUCopyWithDynamicShape from host launch to device launch 

// CHECK-LABEL: func @valid_copy_op_in_replicated_host

// CHECK: "tf_device.launch"
// CHECK: "TPU_REPLICATED_HOST_0"
// CHECK: "tf_device.launch"
// CHECK: "tf.TPUCopyWithDynamicShape"
// CHECK: "TPU_REPLICATED_CORE_0"
func.func @valid_copy_op_in_replicated_host(
  %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
  %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2048xi32>, tensor<2048xi32>) {
  %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
  %0:2 = "tf_device.launch"() ({
        %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
        %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
        %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operand_segment_sizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
        tf_device.return %3#0, %3#1 : tensor<2048xi32>, tensor<2048xi32>
      }) {device = "TPU_REPLICATED_HOST_0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
  return %0#0, %0#1: tensor<2048xi32>, tensor<2048xi32>
}

// CHECK-LABEL: func @valid_copy_op_in_non_replicated_host

// CHECK: "tf_device.launch"
// CHECK: "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK: "tf_device.launch"
// CHECK: "tf.TPUCopyWithDynamicShape"
// CHECK: "/job:localhost/replica:0/task:0/device:TPU:0"
func.func @valid_copy_op_in_non_replicated_host(
  %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
  %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2048xi32>, tensor<2048xi32>) {
  %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
  %0:2 = "tf_device.launch"() ({
        %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
        %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
        %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operand_segment_sizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
        tf_device.return %3#0, %3#1 : tensor<2048xi32>, tensor<2048xi32>
      }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
  return %0#0, %0#1: tensor<2048xi32>, tensor<2048xi32>
}