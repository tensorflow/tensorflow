// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-outside-compiled-to-host-launch | FILECHECK_OPTS="" FileCheck %s

// Tests that outside compilation and model parallelism together fail.
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @outside_compilation_model_parallelism_fail() -> tensor<2xi32> {
    // expected-error@+1 {{outside compilation is not supported with model parallelism}}
    %0 = "tf_device.cluster"() ( {
      %1 = "tf.A"() : () -> tensor<2xi32>
      %2 = "tf.B"(%1) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }) {num_cores_per_replica = 2, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----



module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {

  // Tests that TPU cluster with no outside compilation does not generate launch op.

  // CHECK-LABEL: func @no_outside_compilation
  // CHECK-NOT: "tf_device.launch"
  func @no_outside_compilation() -> tensor<?xi32> {
    %0 = "tf_device.cluster"() ( {
      %1 = "tf.A"() : () -> tensor<?xi32>
      %2 = "tf.B"(%1) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }) {num_cores_per_replica = 1, topology = "", device_assignment = []} : () -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }


  // Tests the launch wrap of a single outside compiled cluster with no input or output dependencies.

  // CHECK-LABEL: func @nodep_single_outside_compilation
  func @nodep_single_outside_compilation() -> () {
    // CHECK:      "tf.A"
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    // CHECK: device_assignment =  [], num_cores_per_replica = 1 : i64, topology =  ""
    "tf_device.cluster"() ( {
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology = "", device_assignment = []} : () -> ()
    return
  }

  // Tests the launch wrap of a single outside compiled cluster with data parallelism.

  // CHECK-LABEL: func @single_outside_compilation_with_replicate
  func @single_outside_compilation_with_replicate(%arg0: tensor<?xi32>) -> () {
    // CHECK:      "tf.A"
    // CHECK:      tf_device.replicate
    // CHECK-NEXT:   "tf_device.cluster"
    // CHECK-NEXT:     "tf.B"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-NEXT:       "tf.C"
    // CHECK-NOT:        _xla_outside_compilation
    // CHECK:            tf_device.return
    // CHECK-NEXT:     device = "TPU_REPLICATED_HOST"
    // CHECK: device_assignment =  [], num_cores_per_replica = 1 : i64, topology =  ""
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      "tf_device.cluster"() ( {
        "tf.B"() : () -> ()
        "tf.C"(%ri_0) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
        "tf.D"() : () -> ()
        tf_device.return
      }) {num_cores_per_replica = 1, topology = "", device_assignment = []} : () -> ()
      tf_device.return
    }
    return
  }

  // Tests launch wrap of a single outside compiled cluster with input/output.

  // CHECK-LABEL: func @single_outside_compilation_input_output
  func @single_outside_compilation_input_output(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:          "tf_device.cluster"
    // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]])
    // CHECK:            tf_device.return %[[B_OUTPUT]]
    // CHECK:          "tf.C"(%[[LAUNCH_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology = "", device_assignment = []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests launch wrap of multiple outside compiled cluster with input/output.

  // CHECK-LABEL: func @multiple_outside_compilation_input_output
  func @multiple_outside_compilation_input_output(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:          "tf_device.cluster"
    // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]])
    // CHECK:            tf_device.return %[[B_OUTPUT]]
    // CHECK:          %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[LAUNCH_OUTPUT]])
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT2:[0-9]*]] = "tf_device.launch"
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[C_OUTPUT]])
    // CHECK:            tf_device.return %[[D_OUTPUT]]
    // CHECK:          %[[LAUNCH_OUTPUT3:[0-9]*]] = "tf_device.launch"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[LAUNCH_OUTPUT2]])
    // CHECK:            tf_device.return %[[E_OUTPUT]]
    // CHECK:          "tf.F"(%[[LAUNCH_OUTPUT3]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        %6 = "tf.D"(%5) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> tensor<?xi32>
        %7 = "tf.E"(%6) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> tensor<?xi32>
        %8 = "tf.F"(%7) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %8 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology = "", device_assignment = []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }
}
