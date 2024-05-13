// RUN: tf-opt %s -split-input-file -tf-xla-rewrite | FileCheck %s


module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  // CHECK-LABEL: func.func @convert_cluster_func
  func.func @convert_cluster_func(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK: "tf.XlaLaunch"(%arg0) <{function = @func, operandSegmentSizes = array<i32: 0, 1, 0>}> : (tensor<i32>) -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg0) {func = @func} : (tensor<i32>) -> tensor<i32>
    func.return %0 : tensor<i32>
  }

  func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0 : tensor<i32>
  }
}

// -----

module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  // CHECK-LABEL: func.func @convert_cluster_func_with_resources_in_order
  func.func @convert_cluster_func_with_resources_in_order(%arg0: tensor<!tf_type.resource>, %arg1: tensor<i32>) -> tensor<i32> {
    // CHECK: "tf.XlaLaunch"(%arg1, %arg0) <{function = @func_with_resources_in_order, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>) -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg1, %arg0) {func = @func_with_resources_in_order} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<i32>)
    func.return %0 : tensor<i32>
  }

  func.func @func_with_resources_in_order(%arg0 : tensor<i32>, %arg1 : tensor<!tf_type.resource>) -> tensor<i32> {
    func.return %arg0 : tensor<i32>
  }
}

// -----

module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  // CHECK-LABEL: func.func @convert_cluster_func_with_resources
  func.func @convert_cluster_func_with_resources(%arg0: tensor<!tf_type.resource>, %arg1: tensor<i32>) -> tensor<i32> {
    // CHECK: "tf.XlaLaunch"(%arg1, %arg0) <{function = @func_with_resources, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>) -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_resources} : (tensor<!tf_type.resource>, tensor<i32>) -> tensor<i32>
    // CHECK: "tf.XlaLaunch"(%arg1, %arg0) <{function = @func_with_resources, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>) -> tensor<i32>
    %1 = "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_resources} : (tensor<!tf_type.resource>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }

  // CHECK-LABEL: func.func @func_with_resources
  // CHECK-SAME:  (%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource>) -> tensor<i32>
  // CHECK:         return %arg0 : tensor<i32>
  func.func @func_with_resources(%arg0 : tensor<!tf_type.resource>, %arg1: tensor<i32>) -> tensor<i32> {
    func.return %arg1 : tensor<i32>
  }
}
