// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-outside-compilation | FILECHECK_OPTS="" FileCheck %s

// Tests that missing `_xla_outside_compilation` attribute value results in an error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // Tests that TPU cluster with no outside compilation does not generate parallel_execute.

  // CHECK-LABEL: func @no_outside_compilation
  func @no_outside_compilation() -> tensor<?xi32> {
    %0 = "tf_device.cluster"() ( {
      %1 = "tf.A"() : () -> tensor<?xi32>
      %2 = "tf.B"(%1) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  // CHECK-NOT: "tf_device.parallel_execute"

  // Tests extraction of a single outside compiled cluster with no input or output dependencies.

  // CHECK-LABEL: func @nodep_single_outside_compilation
  func @nodep_single_outside_compilation() -> () {
     // CHECK: "tf_device.parallel_execute"
     // CHECK-NEXT: "tf_device.launch"
     // CHECK-NEXT: "tf.B"
     // CHECK-NOT: _xla_outside_compilation
     // CHECK-NEXT:   tf_device.return
     // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
     // CHECK: "tf_device.cluster"
     // CHECK-NEXT: "tf.A"
     // CHECK: device_assignment =  [], num_cores_per_replica = 1 : i64, topology =  ""
    "tf_device.cluster"() ( {
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    return
  }

  // Tests extraction of a single outside compiled cluster with multiple ops and no input or output dependecies.

  // CHECK-LABEL: func @nodep_single_cluster_multiple_ops_outside_compilation
  func @nodep_single_cluster_multiple_ops_outside_compilation() -> () {
     // CHECK: "tf_device.parallel_execute"
     // CHECK-NEXT: "tf_device.launch"
     // CHECK-NEXT: "tf.B"
     // CHECK-NEXT: "tf.C"
     // CHECK-NEXT: "tf.D"
     // CHECK-NOT: _xla_outside_compilation
     // CHECK: "tf_device.cluster"
     // CHECK-NEXT: "tf.A"
     // CHECK-NEXT: "tf.E"
     // CHECK: device_assignment =  [], num_cores_per_replica = 1 : i64, topology =  ""
    "tf_device.cluster"() ( {
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.E"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    return
  }

  // Tests extraction of a multiple outside compiled clusters with no input or output dependecies.

  // CHECK-LABEL: func @nodep_multiple_outside_compilation
  func @nodep_multiple_outside_compilation() -> () {
     // CHECK:      "tf_device.parallel_execute"
     // CHECK:      "tf_device.launch"
     // CHECK:        "tf.B"
     // CHECK-NEXT:   "tf.D"
     // CHECK-NOT   "tf_device.launch"
     // CHECK: "tf_device.cluster"
    "tf_device.cluster"() ( {
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      "tf.D"() {_xla_outside_compilation = "cluster2"} : () -> ()
      "tf.E"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    return
  }

  // Tests extraction of a single outside compiled cluster with single TPU cluster return.

  // CHECK-LABEL: func @single_tpu_return_single_outside_compilation
  func @single_tpu_return_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            "tf.B"
    // CHECK-NEXT:       tf_device.return
    // CHECK-NEXT:     device = "TPU_REPLICATED_HOST"
    // CHECK:          %[[TPU_CLUSTER_OUTPUT:[0-9]*]] = "tf_device.cluster"
    // CHECK:            tf_device.return
    // CHECK:          tf_device.return %[[TPU_CLUSTER_OUTPUT]]
    // CHECK:        tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        "tf.A"() : () -> ()
        "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
        %3 = "tf.C"() : () -> tensor<?xi32>
        tf_device.return %3 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster with multiple TPU cluster return.

  // CHECK-LABEL: func @multiple_tpu_return_single_outside_compilation
  func @multiple_tpu_return_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xf32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2  = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:          %[[TPU_CLUSTER_OUTPUT:[0-9]*]]:2 = "tf_device.cluster"
    // CHECK:            tf_device.return
    // CHECK:          tf_device.return %[[TPU_CLUSTER_OUTPUT]]
    // CHECK:        tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
    %1:4 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2, %3 = "tf_device.cluster"() ( {
        %4 = "tf.A"() : () -> tensor<?xf32>
        "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
        %5 = "tf.C"() : () -> tensor<?xi32>
        tf_device.return %4, %5  : tensor<?xf32>, tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> (tensor<?xf32>, tensor<?xi32>)
      tf_device.return %2, %3 : tensor<?xf32>, tensor<?xi32>
    }

    return %1 : tensor<?xf32>
  }

  // Tests extraction of a single outside compiled cluster with single device->host input.

  // CHECK-LABEL: func @single_outside_compiled_input_single_outside_compilation
  func @single_outside_compiled_input_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:[a-z_0-9]+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests value is added as operand to XlaHostCompute op only if defining op is
  // in TPU cluster.

  // CHECK-LABEL: func @single_outside_compiled_input_from_outside_device_cluster
  func @single_outside_compiled_input_from_outside_device_cluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK-NEXT:   %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            "tf.B"(%[[A_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.C"()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %2 = "tf_device.cluster"() ( {
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster with single host->device output.

  // CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation_not_replicated
  func @single_outside_compiled_output_single_outside_compilation_not_replicated(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-NOT:        "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
    // CHECK-SAME:       device_ordinal = 0
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
    %0 = "tf_device.cluster"() ( {
      %1 = "tf.A"() : () -> (tensor<?xi32>)
      %2 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<?xi32>)
      %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %3 : tensor<?xi32>
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>

    return %0 : tensor<?xi32>
  }

  // CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation_replicated
  func @single_outside_compiled_output_single_outside_compilation_replicated(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK:            "tf._XlaSendFromHostV2"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:         "tf_device.cluster"
    // CHECK:           %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:           %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK-SAME:      recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:      send_key = "host_compute_channel_0_args"
    // CHECK:           "tf.C"(%[[HOST_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<?xi32>)
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster host output returned by TPU cluster.

  // CHECK-LABEL: func @return_host_output_outside_compilation
  func @return_host_output_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            tf_device.return %[[HOST_OUTPUT]]
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %5 = "tf.C"(%3) : (tensor<?xi32>) -> (tensor<?xi32>)
        tf_device.return %4 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster with single input/output.

  // CHECK-LABEL: func @single_outside_compiled_input_output_single_outside_compilation
  func @single_outside_compiled_input_output_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests host to device communcation is added only if value is used for ops
  // that are not outside compiled.

  // CHECK-LABEL: func @single_outside_compiled_output_used_for_another_host_op
  func @single_outside_compiled_output_used_for_another_host_op(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-NEXT:       %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK-NEXT:       "tf.IfRegion"(%[[A_OUTPUT]])
    // CHECK-NEXT:         "tf.D"(%[[B_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.C"()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %3 = "tf.A"() : () -> (tensor<i1>)
      %2 = "tf_device.cluster"() ( {
        %4 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<?xi32>)
        "tf.IfRegion"(%3) ({
          "tf.D"(%4) : (tensor<?xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { _xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.C"() : () -> (tensor<?xi32>)
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }


  // Tests extraction of a single outside compiled cluster with multiple input/output.

  // CHECK-LABEL: func @multiple_outside_compiled_input_output_single_outside_compilation
  func @multiple_outside_compiled_input_output_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[B_OUTPUT:[0-9]*]]:2 = "tf.C"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[B_OUTPUT]]#0, %[[B_OUTPUT]]#1, %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]]:2 = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]], %[[B_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            "tf.D"(%[[HOST_OUTPUT]]#0)
    // CHECK:            "tf.E"(%[[HOST_OUTPUT]]#1)
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %5, %6 = "tf.C"(%3, %4) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
        %7 = "tf.D"(%5) : (tensor<?xi32>) -> tensor<?xi32>
        %8 = "tf.E"(%6) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %8 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a multiple outside compiled clusters with input/output.

  // CHECK-LABEL: func @outside_compiled_input_output_multiple_outside_compilation
  func @outside_compiled_input_output_multiple_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT1:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT1]])
    // CHECK:            %[[RECV_OUTPUT2:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[RECV_OUTPUT2]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_1_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT1:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[HOST_OUTPUT1]])
    // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[C_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_1_retvals"
    // CHECK:            "tf.E"(%[[HOST_OUTPUT2]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> (tensor<?xi32>)
        %6 = "tf.D"(%5) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %7 = "tf.E"(%6) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %7 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster with arg input and single device->host input.

  // CHECK-LABEL: func @mixed_input_single_outside_compilation
  func @mixed_input_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%arg0, %[[RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        "tf.B"(%arg0, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a multiple outside compiled clusters with single device->host input.

  // CHECK-LABEL: func @single_outside_compiled_input_multiple_outside_compilation
  func @single_outside_compiled_input_multiple_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT_1:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT_1]])
    // CHECK:            %[[RECV_OUTPUT_2:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_1_args"
    // CHECK:           "tf.D"(%[[RECV_OUTPUT_2]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[C_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_1_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<?xi32>
        "tf.D"(%4) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> ()
        tf_device.return %4 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster with multiple device->host inputs.

  // CHECK-LABEL: func @multiple_outside_compiled_inputs_single_outside_compilation
  func @multiple_outside_compiled_inputs_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT_1:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.C"(%[[RECV_OUTPUT_1]])
    // CHECK:            %[[RECV_OUTPUT_2:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_1_args"
    // CHECK:            "tf.D"(%[[RECV_OUTPUT_2]], %[[RECV_OUTPUT_1]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[B_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_1_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        "tf.C"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
        "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> ()
        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests only directly used results of tpu cluster are remapped with
  // parallel_execute.

  // CHECK-LABEL: func @remapped_results
  func @remapped_results(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:   %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
    // CHECK: tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]#1 : tensor<?xi32>
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2:2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %5:2 = "tf.C"(%4) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
        tf_device.return %5#0, %5#1 : tensor<?xi32>, tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> (tensor<?xi32>, tensor<?xi32>)
      tf_device.return %2#1 : tensor<?xi32>
    }
    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if
  func @outside_compiled_ops_inside_tf_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK-NEXT:         %[[ARG_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK:              "tf.D"(%[[ARG_RECV_OUTPUT]]#0, %[[ARG_RECV_OUTPUT]]#1)
    // CHECK-NOT:          "tf._XlaSendFromHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:              "tf.Yield"() : () -> ()
    // CHECK:            _else_func_name = "test_else_name"
    // CHECK-SAME        _then_func_name = "test_then_name"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:              "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK-SAME:         tpu_core = 0
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false, _then_func_name = "test_then_name", _else_func_name = "test_else_name"} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Ensures that separate send/recvs are added for values that are used by ops inside of multiple IfRegions.

  // CHECK-LABEL: func @outside_compiled_ops_multiple_tf_if
  func @outside_compiled_ops_multiple_tf_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:           %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_2"
    // CHECK:              %[[ARG_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK:              "tf.D"(%[[ARG_RECV_OUTPUT]])
    // CHECK:              %[[ARG_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_1_args"
    // CHECK:              "tf.F"(%[[ARG_RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            "tf._XlaHostComputeMlir"
    // CHECK-SAME:       key = "if_predicate_channel_2"
    // CHECK-NEXT:       tf.IfRegion"
    // CHECK:              "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK:            tf.IfRegion"
    // CHECK:              "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_1_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_1_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%3) {_xla_outside_compilation = "auto"} : (tensor<?xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        "tf.IfRegion"(%6) ({
          "tf.F"(%3) {_xla_outside_compilation = "auto"} : (tensor<?xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        tf_device.return %3 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled

  // CHECK-LABEL: func @outside_compiled_tf_if
  func @outside_compiled_tf_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK-SAME:       (tensor<3x!tf.string>, tensor<i64>) -> (tensor<?xi32>, tensor<?xi32>, tensor<i1>)
    // CHECK-NEXT:       tf.IfRegion"(%[[RECV_OUTPUT]]#2)
    // CHECK:              "tf.D"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1, %[[F_OUT]])
    // CHECK-NOT:          "tf._XlaSendFromHostV2"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]], %[[G_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK-SAME:       tpu_core = 0
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %7 = "tf.F"() : () -> tensor<?xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%4, %3, %7) {} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled and wrapped inside another
  // tf.IfRegion op

  // CHECK-LABEL: func @outside_compiled_tf_if_nested
  func @outside_compiled_tf_if_nested(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT_PREDICATE:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-SAME:       (tensor<3x!tf.string>, tensor<i64>) -> tensor<i1>
    // CHECK-NEXT:       tf.IfRegion"(%[[RECV_OUTPUT_PREDICATE]])
    // CHECK-NEXT:         %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK-SAME:         (tensor<3x!tf.string>, tensor<i64>) -> (tensor<?xi32>, tensor<i1>)
    // CHECK-NEXT:         tf.IfRegion"(%[[RECV_OUTPUT]]#1)
    // CHECK-NEXT:           "tf.H"(%[[RECV_OUTPUT]]#0, %[[F_OUT]])
    // CHECK:                "tf.Yield"() : () -> ()
    // CHECK:                "tf.Yield"() : () -> ()
    // CHECK-NOT:          "tf._XlaSendFromHostV2"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[G_OUTPUT]])
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-SAME:       (tensor<i1>) -> ()
    // CHECK-NEXT:       "tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:              %[[D_OUT:[0-9]*]] = "tf.D"
    // CHECK-NEXT:         %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:              "tf._XlaHostComputeMlir"(%[[D_OUT]], %[[F_OUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK-SAME:         tpu_core = 0
    // CHECK:              "tf.Yield"() : () -> ()
    // CHECK:              "tf.Yield"() : () -> ()
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %7 = "tf.F"() : () -> tensor<?xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %8 = "tf.D"(%4, %3, %7) {} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
          %9 = "tf.F"(%4) {} : (tensor<?xi32>) -> (tensor<i1>)

          "tf.IfRegion"(%9) ({
            "tf.H"(%8, %7) : (tensor<?xi32>, tensor<?xi32>) -> ()
            "tf.Yield"() : () -> ()
          }, {
            "tf.Yield"() : () -> ()
          }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion
  // op with return values.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if_with_return_values
  func @outside_compiled_ops_inside_tf_if_with_return_values(
    %arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK-NEXT:         %[[ARG_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK:              %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[ARG_RECV_OUTPUT]]#0, %[[ARG_RECV_OUTPUT]]#1)
    // CHECK:              "tf._XlaSendFromHostV2"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_retvals"
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%6)
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:              %[[HOST_COMPUTE_OUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK-SAME:         tpu_core = 0
    // CHECK-NEXT:         "tf.Yield"(%[[HOST_COMPUTE_OUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %7 = "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
          "tf.Yield"(%7) : (tensor<?xi32>) -> ()
        }, {

          %8 = "tf.F"() : () -> (tensor<?xi32>)
          "tf.Yield"(%8) : (tensor<?xi32>) -> ()
        }) { is_stateless = false} : (tensor<i1>) -> (tensor<?xi32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op without external inputs/outputs

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if_without_input_outputs
  func @outside_compiled_ops_inside_tf_if_without_input_outputs(
    %arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_0"
    // CHECK-NEXT:       tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK:              "tf.D"
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%6)
    // CHECK-SAME:       key = "if_predicate_channel_0"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a nested
  // tf.IfRegion op.

  // CHECK-LABEL: func @outside_compiled_ops_inside_nested_if
  func @outside_compiled_ops_inside_nested_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_2"
    // CHECK-NEXT:      tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK-NEXT:        %[[PREDICATE2_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:        key = "if_predicate_channel_1"
    // CHECK-NEXT:        tf.IfRegion"(%[[PREDICATE2_RECV_OUTPUT]])
    // CHECK-NEXT:          "tf.Yield"() : () -> ()
    // CHECK:               %[[ARG_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:          key = "host_compute_channel_0_args"
    // CHECK:               "tf.D"(%[[ARG_RECV_OUTPUT]])
    // CHECK-NOT:           "tf._XlaSendFromHostV2"
    // CHECK-NEXT:          "tf.Yield"() : () -> ()

    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[G_OUTPUT]])
    // CHECK-SAME        key = "if_predicate_channel_2"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"(%[[B_OUTPUT]])
    // CHECK:              "tf._XlaHostComputeMlir"(%[[H_OUTPUT]])
    // CHECK-SAME:         key = "if_predicate_channel_1"
    // CHECK-NEXT:         tf.IfRegion"(%[[H_OUTPUT]])
    // CHECK-NEXT:           "tf.Yield"() : () -> ()
    // CHECK:                 %[[I_OUTPUT:[0-9]*]] = "tf.I"(%[[H_OUTPUT]])
    // CHECK:                 "tf._XlaHostComputeMlir"(%[[I_OUTPUT]])
    // CHECK-NEXT:            "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
           %7 = "tf.H"(%4) : (tensor<?xi32>) -> (tensor<i1>)

          "tf.IfRegion"(%7)({
              "tf.Yield"() : () -> ()
            },
            {
              %8 = "tf.I"(%7) : (tensor<i1>) -> (tensor<?xi32>)
              "tf.D"(%8) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
              "tf.Yield"() : () -> ()
            }) { is_stateless = false} : (tensor<i1>) -> ()

          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.WhileRegion op body.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_body
  func @outside_compiled_ops_inside_tf_while_body(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       tf.WhileRegion"
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "while_condition_channel_0"
    // CHECK:              "tf.Yield"(%[[COND_RECV_OUTPUT]])
    // CHECK:              %[[BODY_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:              %[[D_OUTPUT:[0-9]*]] = "tf.D"
    // CHECK:              "tf._XlaSendFromHostV2"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         "tf.Yield"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK-NEXT:       tf.WhileRegion"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUTPUT]])
    // CHECK:              %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK-NEXT:         %[[HOST_COMPUTE_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"
    // CHECK-NEXT:         "tf.Yield"(%[[C_OUTPUT]], %[[HOST_COMPUTE_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %9 = "tf.D"(%arg1, %arg2) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
          "tf.Yield"(%8, %9) : (tensor<i32>, tensor<?xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.WhileRegion op cond.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_cond
  func @outside_compiled_ops_inside_tf_while_cond(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       tf.WhileRegion"
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT1:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         %[[I_OUTPUT:[0-9]*]] = "tf.I"(%[[COND_RECV_OUTPUT1]]#0, %[[COND_RECV_OUTPUT1]]#1)
    // CHECK-NEXT:         "tf._XlaSendFromHostV2"(%[[I_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT2:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "while_condition_channel_0"
    // CHECK:              "tf.Yield"(%[[COND_RECV_OUTPUT2]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK-NEXT:       tf.WhileRegion"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:              %[[HOST_COMPUTE_OUTPUT:.+]] = "tf._XlaHostComputeMlir"
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"(%[[HOST_COMPUTE_OUTPUT]])
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUTPUT]])
    // CHECK:              %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK-NEXT:         %[[D_OUTPUT:.+]] = "tf.D"
    // CHECK-NEXT:         "tf.Yield"(%[[C_OUTPUT]], %[[D_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.I"(%arg1, %arg2) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<?xf32>) -> tensor<i32>
          %8 = "tf.H"(%7) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%8) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %8 = "tf.D"(%arg1, %arg2) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
          "tf.Yield"(%7, %8) : (tensor<i32>, tensor<?xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.WhileRegion op cond and body.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_cond_body
  func @outside_compiled_ops_inside_tf_while_cond_body(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       tf.WhileRegion"
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT1:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         %[[I_OUTPUT:[0-9]*]] = "tf.I"(%[[COND_RECV_OUTPUT1]]#0, %[[COND_RECV_OUTPUT1]]#1)
    // CHECK-NEXT:         "tf._XlaSendFromHostV2"(%[[I_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "while_condition_channel_0"
    // CHECK:              "tf.Yield"(%[[COND_RECV_OUTPUT]])
    // CHECK:              %[[BODY_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:              %[[D_OUTPUT:[0-9]*]] = "tf.D"
    // CHECK:              "tf._XlaSendFromHostV2"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         "tf.Yield"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK-NEXT:       tf.WhileRegion"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUTPUT]])
    // CHECK:              %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:              %[[HOST_COMPUTE_OUTPUT:.+]] = "tf._XlaHostComputeMlir"
    // CHECK-NEXT:         "tf.Yield"(%[[C_OUTPUT]], %[[HOST_COMPUTE_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.I"(%arg1, %arg2) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<?xf32>) -> tensor<i32>
          %8 = "tf.H"(%7) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%8) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %8 = "tf.D"(%arg1, %arg2) {_xla_outside_compilation = "cluster2"} : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
          "tf.Yield"(%7, %8) : (tensor<i32>, tensor<?xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op
  // nested in a tf.WhileRegion.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_if
  func @outside_compiled_ops_inside_tf_while_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       tf.WhileRegion"
    // CHECK-NEXT:         %[[COND_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "while_condition_channel_1"
    // CHECK:              "tf.Yield"(%[[COND_RECV_OUTPUT]])
    // CHECK:              %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:         tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK:                %[[D_OUTPUT:[0-9]*]] = "tf.D"
    // CHECK:                "tf._XlaSendFromHostV2"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:           "tf.Yield"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK-NEXT:       tf.WhileRegion"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUTPUT]])
    // CHECK:              %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK-NEXT:         "tf._XlaHostComputeMlir"(%[[G_OUTPUT]])
    // CHECK-NEXT:         tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK-NEXT:         %[[HOST_COMPUTE_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"
    // CHECK-NEXT:         "tf.Yield"(%[[HOST_COMPUTE_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> tensor<?xf32>
            "tf.Yield"(%9) : (tensor<?xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<?xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<?xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<?xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled with a nested tf.WhileRegion op.

  // CHECK-LABEL: func @outside_compiled_tf_if_nested_while
  func @outside_compiled_tf_if_nested_while(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK-SAME:       (tensor<3x!tf.string>, tensor<i64>) -> (tensor<?xi32>, tensor<?xi32>, tensor<i1>)
    // CHECK-NEXT:       tf.IfRegion"(%[[RECV_OUTPUT]]#2)
    // CHECK:              %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1, %[[F_OUT]])
    // CHECK-NEXT:         %[[J_OUTPUT:[0-9]*]] = "tf.J"
    // CHECK-NEXT:         %[[K_OUTPUT:[0-9]*]] = "tf.K"
    // CHECK-NEXT:          tf.WhileRegion"(%[[J_OUTPUT]], %[[D_OUTPUT]])
    // CHECK:                 %[[H_OUTPUT:[0-9]*]] = "tf.H"(%[[K_OUTPUT]])
    // CHECK-NOT:           "tf._XlaSendFromHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]], %[[G_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK-SAME:       tpu_core = 0
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %7 = "tf.F"() : () -> tensor<?xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %8 = "tf.D"(%4, %3, %7) {} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?xf32>)
          %9 = "tf.J"() : () -> (tensor<i32>)
          %10 = "tf.K"() : () -> (tensor<i32>)
          "tf.WhileRegion"(%9, %8) ({
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
            %11 = "tf.I"(%arg1, %arg2) : (tensor<i32>, tensor<?xf32>) -> tensor<i32>
            %12 = "tf.H"(%10) :  (tensor<i32>) -> tensor<i1>
            "tf.Yield"(%12) : (tensor<i1>) -> ()
          }, {
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
            %11 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
            %12 = "tf.D"(%arg1, %arg2) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
            "tf.Yield"(%11, %12) : (tensor<i32>, tensor<?xf32>) -> ()
          }) { is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of an outside compiled tf.WhileRegion where the entire
  // tf.WhileRegion op is outside compiled with a nested tf.IfRegion.

  // CHECK-LABEL: func @outside_compiled_ops_tf_while_nested_if
  func @outside_compiled_ops_tf_while_nested_if(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[HOST_RECV_OUTPUT:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:           "tf.WhileRegion"(%[[HOST_RECV_OUTPUT]]#1, %[[HOST_RECV_OUTPUT]]#2)
    // CHECK:             %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:             "tf.IfRegion"(%[[HOST_RECV_OUTPUT]]#0)
    // CHECK:               %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[C_OUTPUT]])
    // CHECK-NEXT:          "tf.Yield"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[G_OUTPUT]], %[[B_OUTPUT]], %[[A_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"(%8) : (tensor<i32>) -> tensor<?xf32>
            "tf.Yield"(%9) : (tensor<?xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<?xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<?xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<?xf32>) -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Tests extraction of an outside compiled cluster that contains ops wrapped
  // inside multiple regions of nested tf.IfRegion and tf.WhileRegion.

  // CHECK-LABEL: func @outside_compiled_ops_with_multiple_region_single_cluster
  func @outside_compiled_ops_with_multiple_region_single_cluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._TPUCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      "tf._XlaSendFromHostV2"(%[[B_OUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:      "tf.WhileRegion"()
    // CHECK-NEXT:        %[[WHILE_COND:.*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:        "tf.Yield"(%[[WHILE_COND]])
    // CHECK:             %[[C_OUT:.*]] = "tf.C"(%[[B_OUT]])
    // CHECK-NEXT:        "tf._XlaSendFromHostV2"(%[[C_OUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:        %[[IF_COND:.*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-NEXT:        "tf.IfRegion"(%[[IF_COND]])
    // CHECK-NEXT:           %[[D_OUT:.*]] = "tf.D"(%[[C_OUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:       %[[B_OUT_DEVICE:.*]] = "tf._XlaHostComputeMlir"()
    // CHECK-NEXT:       %[[G_OUT:.*]] = "tf.G"
    // CHECK-NEXT:       "tf.WhileRegion"(%[[B_OUT_DEVICE]], %[[A_OUT]])
    // CHECK:              %[[H_OUT:.*]] = "tf.H"
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUT]])
    // CHECK:              "tf._XlaHostComputeMlir"(%[[G_OUT]])
    // CHECK-NEXT:         "tf.IfRegion"(%[[G_OUT]])
    // CHECK-NEXT:           "tf._XlaHostComputeMlir"()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> (tensor<?xf32>)
        %4 = "tf.B"() {_xla_outside_compilation="cluster0"} : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)
        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
          %8 = "tf.C"(%4) {_xla_outside_compilation="cluster0"} : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"(%8) {_xla_outside_compilation="cluster0"} : (tensor<i32>) -> tensor<?xf32>
            "tf.Yield"(%9) : (tensor<?xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<?xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<?xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<?xf32>) -> ()
        }) {is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)

        %5 = "tf.E"() : () -> tensor<?xi32>
        tf_device.return %5 : tensor<?xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }

    return %1 : tensor<?xi32>
  }

  // Verifies that ops in between outside compile ops and depending on results
  // from the host are moved after the host compute op so that dominance is not
  // violated. tf.C op in this case.
  // CHECK-LABEL: func @device_op_dominance
  func @device_op_dominance() -> () {
    // CHECK: tf.B
    // CHECK: tf._XlaSendFromHost
    // CHECK: tf._XlaRecvAtHost
    // CHECK: tf.D

    // CHECK: tf.A
    // CHECK: tf._XlaHostComputeMlir
    // CHECK: tf.C
    // CHECK: tf.E

    "tf_device.cluster"() ( {
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"() {_xla_outside_compilation = "cluster0"} : () -> (tensor<i32>)
      "tf.C"(%1) : (tensor<i32>) -> ()
      "tf.D"(%1, %0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.E"(%0, %1) : (tensor<i32>, tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    return
  }

  // Verifies that ops indirectly depending on results from the host are also
  // moved after the host compute op. tf.E op in this case.

  // CHECK-LABEL: func @device_op_dominance_with_indirect_dependency
  func @device_op_dominance_with_indirect_dependency() -> () {
    // CHECK: tf.B
    // CHECK: tf._XlaRecvAtHost
    // CHECK: tf.F

    // CHECK: tf.A
    // CHECK: tf.C
    // CHECK: tf.D
    // CHECK: tf.E
    // CHECK: tf._XlaHostComputeMlir
    // CHECK: tf.G

    "tf_device.cluster"() ( {
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"() {_xla_outside_compilation = "cluster0"} : () -> (tensor<i32>)
      %2 = "tf.C"(%1) : (tensor<i32>) -> (tensor<i32>)
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.E"(%2, %3) : (tensor<i32>, tensor<i32>) -> ()
      "tf.F"(%1, %0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.G"(%0, %1) : (tensor<i32>, tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    return
  }
}
