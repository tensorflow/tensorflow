// RUN: tf-opt %s -split-input-file -tf-xla-rewrite-v2 | FileCheck %s


module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  // CHECK-LABEL: func.func @convert_cluster_func
  func.func @convert_cluster_func(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaCompile"(%arg0) <{function = @func, must_compile = true, operandSegmentSizes = array<i32: 0, 1, 0>}> : (tensor<i32>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaRun"(%arg0, %0#0) : (tensor<i32>, tensor<3x!tf_type.string>) -> tensor<i32>
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
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) <{function = @func_with_resources_in_order, must_compile = true, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>)
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %0#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
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
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) <{function = @func_with_resources_1, must_compile = true, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %0#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
    %0 = "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_resources_1, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<!tf_type.resource>, tensor<i32>) -> tensor<i32>
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaCompile"(%arg1, %arg0) <{function = @func_with_resources_2, must_compile = true, operandSegmentSizes = array<i32: 0, 1, 1>}> : (tensor<i32>, tensor<!tf_type.resource>) -> (tensor<3x!tf_type.string>, tensor<!tf_type.boolref>)
    // CHECK: "tf_device.launch"()
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaRun"(%arg1, %arg0, %2#0) : (tensor<i32>, tensor<!tf_type.resource>, tensor<3x!tf_type.string>) -> tensor<i32>
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

// -----

// CHECK-LABEL: func.func @outside_compilation_in_generic_pipeline
module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0"], tf.versions = {producer = 888 : i32}} {
  func.func @outside_compilation_in_generic_pipeline(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK: tf_device.launch
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: "tf._XlaCompile"() <{function = @func, must_compile = true, operandSegmentSizes = array<i32: 0, 0, 0>}>
    // CHECK: tf_device.parallel_execute
    // CHECK: tf_device.launch
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
    // CHECK: tf.B
    // CHECK: tf._XlaSendFromHost
    // CHECK: tf_device.launch
    // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:GPU:0"}>
    // CHECK: tf._XlaRun
    %0 = "tf_device.parallel_execute"() ({
      "tf_device.launch"() ({
        %1 = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        %2 = "tf.B"() : () -> tensor<2xi32>
        "tf._XlaSendFromHost"(%2, %1) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_0_retvals"} : (tensor<2xi32>, tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
      tf_device.return
    }, {
      %0 = "tf_device.cluster_func"() {func = @func, device = "/job:localhost/replica:0/task:0/device:GPU:0"} : () -> tensor<2xi32>
      tf_device.return %0 : tensor<2xi32>
    }) : () -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
  func.func @func() -> tensor<2xi32> {
    %2 = "tf.A"() : () -> tensor<2xi32>
    %3 = "tf._XlaHostComputeMlir"() {host_mlir_module = "", manual_sharding = false, recv_key = "host_compute_channel_0_retvals", send_key = "host_compute_channel_0_args"} : () -> tensor<2xi32>
    %4 = "tf.C"(%3) : (tensor<2xi32>) -> tensor<2xi32>
    func.return %4 : tensor<2xi32>
  }
}
