// RUN: tf-opt -split-input-file -verify-diagnostics -tf-extract-tpu-copy-with-dynamic-shape-op %s | FileCheck %s

// Test that extract TPUCopyWithDynamicShape from host launch to device launch.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:COMPOSITE:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}} {
  // CHECK-LABEL: func @valid_copy_op_in_replicated_host

  // CHECK: "tf_device.launch"
  // CHECK-SAME: "TPU_REPLICATED_HOST_0"
  // CHECK: "tf_device.launch"
  // CHECK-SAME: "TPU_REPLICATED_CORE_0"
  // CHECK: "tf.TPUCopyWithDynamicShape"
  func.func @valid_copy_op_in_replicated_host(
    %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
    %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2048xi32>, tensor<2048xi32>) {
    %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
    %0:2 = "tf_device.launch"() ({
	  %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
	  %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
	  %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
	  tf_device.return %3#0, %3#1 : tensor<2048xi32>, tensor<2048xi32>
	}) {device = "TPU_REPLICATED_HOST_0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
    return %0#0, %0#1: tensor<2048xi32>, tensor<2048xi32>
  }

  // CHECK-LABEL: func @valid_copy_op_in_non_replicated_host

  // CHECK: "tf_device.launch"
  // CHECK-SAME: "/job:localhost/replica:0/task:0/device:CPU:0"
  // CHECK: "tf_device.launch"
  // CHECK-SAME: "/job:localhost/replica:0/task:0/device:TPU:0"
  // CHECK: "tf.TPUCopyWithDynamicShape"
  func.func @valid_copy_op_in_non_replicated_host(
    %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
    %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2048xi32>, tensor<2048xi32>) {
    %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
    %0:2 = "tf_device.launch"() ({
	  %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
	  %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
	  %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
	  tf_device.return %3#0, %3#1 : tensor<2048xi32>, tensor<2048xi32>
	}) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
    return %0#0, %0#1: tensor<2048xi32>, tensor<2048xi32>
  }

  // CHECK-LABEL: func @copy_and_send

  // CHECK: "tf_device.launch"
  // CHECK: "tf.TPUCopyWithDynamicShape"
  // CHECK: "tf._XlaSendFromHostV2
  // CHECK: tf_device.return
  // CHECK-NOT: launch
  // CHECK: return
  func.func @copy_and_send(%arg0: tensor<65536xi64>, %arg1: tensor<1x!tf_type.string>, %arg2: tensor<65536xi32>) {
    "tf_device.launch"() ({
	%7088 = "tf.TPUCopyWithDynamicShape"(%arg2, %arg2) {operandSegmentSizes = array<i32: 1, 1>} : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi64>
	"tf._XlaSendFromHostV2"(%arg1, %7088) {key = "foo"} : (tensor<1x!tf_type.string>, tensor<65536xi64>) -> ()
	tf_device.return
      }) {device = "TPU_REPLICATED_HOST_0"} : () -> ()
    return
  }
}

// -----

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:COMPOSITE:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}} {
  func.func @bad_host0(
    %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
    %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2048xi32>, tensor<2048xi32>) {
    %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
    %0:2 = "tf_device.launch"() ({
      %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
      %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
      // expected-error @+1 {{device is not a recognized host 0}}
      %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
      tf_device.return %3#0, %3#1 : tensor<2048xi32>, tensor<2048xi32>
    }) {device = "/job:localhost/replica:0/task:1/device:CPU:0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
    return %0#0, %0#1: tensor<2048xi32>, tensor<2048xi32>
  }
}
