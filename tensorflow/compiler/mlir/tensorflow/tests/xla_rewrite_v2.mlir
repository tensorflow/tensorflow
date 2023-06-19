// RUN: tf-opt %s -split-input-file -tf-xla-rewrite-v2 | FileCheck %s


module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  // CHECK-LABEL: func.func @convert_cluster_func
  func.func @convert_cluster_func(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaCompile"(%arg0) {function = @func, must_compile = true, operand_segment_sizes = array<i32: 0, 1, 0>} : (tensor<i32>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"}
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaRun"(%arg0, %0#0) : (tensor<i32>, tensor<3x!tf_type.string>) -> tensor<i32>
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"} : () -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg0) {func = @func, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<i32>) -> tensor<i32>
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
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) {function = @func_with_resources_in_order, must_compile = true, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i32>, tensor<!tf_type.resource>)
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"}
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %0#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"} : () -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg1, %arg0) {func = @func_with_resources_in_order, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<i32>)
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
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) {function = @func_with_resources_1, must_compile = true, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"}
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %0#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"} : () -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_resources_1, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<!tf_type.resource>, tensor<i32>) -> tensor<i32>
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) {function = @func_with_resources_2, must_compile = true, operand_segment_sizes = array<i32: 0, 1, 1>} : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"}
    // CHECK: "tf_device.launch"()
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %2#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
    // CHECK: {device = "/job:localhost/replica:0/task:0/device:GPU:0"} : () -> tensor<i32>
    %1 = "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_resources_2, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<!tf_type.resource>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }


  func.func @func_with_resources_1(%arg0 : tensor<!tf_type.resource>, %arg1: tensor<i32>) -> tensor<i32> {
    func.return %arg1 : tensor<i32>
  }

  func.func @func_with_resources_2(%arg0 : tensor<!tf_type.resource>, %arg1: tensor<i32>) -> tensor<i32> {
    func.return %arg1 : tensor<i32>
  }
}
