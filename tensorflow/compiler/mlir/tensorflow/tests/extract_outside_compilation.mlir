// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-extract-outside-compilation | FILECHECK_OPTS="" FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // Tests that TPU cluster with no outside compilation does not generate parallel_execute.

  // CHECK-LABEL: func @no_outside_compilation
  func.func @no_outside_compilation() -> tensor<2xi32> {
    // CHECK-NOT: "tf_device.parallel_execute"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<2xi32>
      %2 = "tf.B"(%1) : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }

  // CHECK-LABEL: func @attribute_outside_of_cluster
  func.func @attribute_outside_of_cluster() -> tensor<2xi32> {
    // CHECK-NOT: _xla_outside_compilation
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<2xi32>
      tf_device.return %1 : tensor<2xi32>
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
    %2 = "tf.B"(%0) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }


  // Tests extraction of a single outside compiled cluster with no input or output dependencies.

  // CHECK-LABEL: func @nodep_single_outside_compilation
  func.func @nodep_single_outside_compilation() -> () {
     // CHECK: "tf_device.parallel_execute"
     // CHECK-NEXT: "tf_device.launch"
     // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:CPU:0"
     // CHECK-NEXT: "tf.B"
     // CHECK-NOT: _xla_outside_compilation
     // CHECK-NEXT:   tf_device.return
     // CHECK: "tf_device.cluster"
     // CHECK-NEXT: "tf.A"
     // CHECK: device_assignment =  [], num_cores_per_replica = 1 : i64, topology =  ""
    "tf_device.cluster"() ({
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests extraction of a single outside compiled cluster with multiple ops and no input or output dependecies.

  // CHECK-LABEL: func @nodep_single_cluster_multiple_ops_outside_compilation
  func.func @nodep_single_cluster_multiple_ops_outside_compilation() -> () {
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
    "tf_device.cluster"() ({
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.E"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests extraction of a multiple outside compiled clusters with no input or output dependecies.

  // CHECK-LABEL: func @nodep_multiple_outside_compilation
  func.func @nodep_multiple_outside_compilation() -> () {
     // CHECK:      "tf_device.parallel_execute"
     // CHECK:      "tf_device.launch"
     // CHECK:        "tf.B"
     // CHECK-NEXT:   "tf.D"
     // CHECK-NOT   "tf_device.launch"
     // CHECK: "tf_device.cluster"
    "tf_device.cluster"() ({
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      "tf.D"() {_xla_outside_compilation = "cluster2"} : () -> ()
      "tf.E"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests extraction of a single outside compiled cluster with single TPU cluster return.

  // CHECK-LABEL: func @single_tpu_return_single_outside_compilation
  func.func @single_tpu_return_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-SAME:     device = "TPU_REPLICATED_HOST_0"
    // CHECK:            "tf.B"
    // CHECK-NEXT:       tf_device.return
    // CHECK:          %[[TPU_CLUSTER_OUTPUT:[0-9]*]] = "tf_device.cluster"
    // CHECK:            tf_device.return
    // CHECK:          tf_device.return %[[TPU_CLUSTER_OUTPUT]]
    // CHECK:        tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        "tf.A"() : () -> ()
        "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
        %3 = "tf.C"() : () -> tensor<2xi32>
        tf_device.return %3 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster with multiple TPU cluster return.

  // CHECK-LABEL: func @multiple_tpu_return_single_outside_compilation
  func.func @multiple_tpu_return_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<3xf32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2  = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:          %[[TPU_CLUSTER_OUTPUT:[0-9]*]]:2 = "tf_device.cluster"
    // CHECK:            tf_device.return
    // CHECK:          tf_device.return %[[TPU_CLUSTER_OUTPUT]]
    // CHECK:        tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
    %1:4 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2, %3 = "tf_device.cluster"() ({
        %4 = "tf.A"() : () -> tensor<3xf32>
        "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
        %5 = "tf.C"() : () -> tensor<2xi32>
        tf_device.return %4, %5  : tensor<3xf32>, tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> (tensor<3xf32>, tensor<2xi32>)
      tf_device.return %2, %3 : tensor<3xf32>, tensor<2xi32>
    }

    func.return %1 : tensor<3xf32>
  }

  // Tests extraction of a single outside compiled cluster with single device->host input.

  // CHECK-LABEL: func @single_outside_compiled_input_single_outside_compilation
  func.func @single_outside_compiled_input_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:[a-z_0-9]+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       host_mlir_module = ""
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<2xi32>
        tf_device.return %4 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests value is added as operand to XlaHostCompute op only if defining op is
  // in TPU cluster.

  // CHECK-LABEL: func @single_outside_compiled_input_from_outside_device_cluster
  func.func @single_outside_compiled_input_from_outside_device_cluster(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK-NEXT:   %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            "tf.B"(%[[A_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.C"()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %3 = "tf.A"() : () -> (tensor<2xi32>)
      %2 = "tf_device.cluster"() ({
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<2xi32>
        tf_device.return %4 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster with single host->device output.

  // CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation_not_replicated
  func.func @single_outside_compiled_output_single_outside_compilation_not_replicated(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-NOT:        "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
    // CHECK-SAME:       device_ordinal = 0
    // CHECK-SAME:       device_type = "TPU"
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> (tensor<2xi32>)
      %2 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<2xi32>)
      %3 = "tf.C"(%2) : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %3 : tensor<2xi32>
    }) {_xla_compile_device_type = "TPU", num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>

    func.return %0 : tensor<2xi32>
  }

  // CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation_replicated
  func.func @single_outside_compiled_output_single_outside_compilation_replicated(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<2xi32>)
        %5 = "tf.C"(%4) : (tensor<2xi32>) -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster host output returned by TPU cluster.

  // CHECK-LABEL: func @return_host_output_outside_compilation
  func.func @return_host_output_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            tf_device.return %[[B_OUTPUT]]
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            tf_device.return
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %5 = "tf.C"(%3) : (tensor<2xi32>) -> (tensor<2xi32>)
        tf_device.return %4 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster with single input/output.

  // CHECK-LABEL: func @single_outside_compiled_input_output_single_outside_compilation
  func.func @single_outside_compiled_input_output_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %5 = "tf.C"(%4) : (tensor<2xi32>) -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests host to device communcation is added only if value is used for ops
  // that are not outside compiled.

  // CHECK-LABEL: func @single_outside_compiled_output_used_for_another_host_op
  func.func @single_outside_compiled_output_used_for_another_host_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-NEXT:       %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK-NEXT:       "tf.IfRegion"(%[[A_OUTPUT]])
    // CHECK-NEXT:         "tf.D"(%[[B_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.C"()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %3 = "tf.A"() : () -> (tensor<i1>)
      %2 = "tf_device.cluster"() ({
        %4 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<2xi32>)
        "tf.IfRegion"(%3) ({
          "tf.D"(%4) : (tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { _xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.C"() : () -> (tensor<2xi32>)
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }


  // Tests extraction of a single outside compiled cluster with multiple input/output.

  // CHECK-LABEL: func @multiple_outside_compiled_input_output_single_outside_compilation
  func.func @multiple_outside_compiled_input_output_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %5, %6 = "tf.C"(%3, %4) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
        %7 = "tf.D"(%5) : (tensor<2xi32>) -> tensor<2xi32>
        %8 = "tf.E"(%6) : (tensor<2xi32>) -> tensor<2xi32>
        tf_device.return %8 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a multiple outside compiled clusters with input/output.

  // CHECK-LABEL: func @outside_compiled_input_output_multiple_outside_compilation
  func.func @outside_compiled_input_output_multiple_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %5 = "tf.C"(%4) : (tensor<2xi32>) -> (tensor<2xi32>)
        %6 = "tf.D"(%5) {_xla_outside_compilation = "cluster2"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %7 = "tf.E"(%6) : (tensor<2xi32>) -> tensor<2xi32>
        tf_device.return %7 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster with arg input and single device->host input.

  // CHECK-LABEL: func @mixed_input_single_outside_compilation
  func.func @mixed_input_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%arg0, %[[RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        "tf.B"(%arg0, %3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>, tensor<2xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<2xi32>
        tf_device.return %4 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a multiple outside compiled clusters with single device->host input.

  // CHECK-LABEL: func @single_outside_compiled_input_multiple_outside_compilation
  func.func @single_outside_compiled_input_multiple_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT_1:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT_1]])
    // CHECK:            %[[RECV_OUTPUT_2:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_1_args"
    // CHECK:           "tf.D"(%[[RECV_OUTPUT_2]])
    // CHECK:          "tf_device.cluster"()
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[C_OUTPUT]])
    // CHECK-SAME:       send_key = "host_compute_channel_1_args"
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> ()
        %4 = "tf.C"() : () -> tensor<2xi32>
        "tf.D"(%4) {_xla_outside_compilation = "cluster2"} : (tensor<2xi32>) -> ()
        tf_device.return %4 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster with multiple device->host inputs.

  // CHECK-LABEL: func @multiple_outside_compiled_inputs_single_outside_compilation
  func.func @multiple_outside_compiled_inputs_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        "tf.C"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> ()
        "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>, tensor<2xi32>) -> ()
        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests only directly used results of tpu cluster are remapped with
  // parallel_execute.

  // CHECK-LABEL: func @remapped_results
  func.func @remapped_results(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:   %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
    // CHECK: tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]#1 : tensor<2xi32>
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2:2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %5:2 = "tf.C"(%4) : (tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
        tf_device.return %5#0, %5#1 : tensor<2xi32>, tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> (tensor<2xi32>, tensor<2xi32>)
      tf_device.return %2#1 : tensor<2xi32>
    }
    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion
  // op. Check that we mark the rewritten control-flow as stateful even though
  // the original control-flow is not (this is due to added side-effecting
  // communication ops).

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if
  func.func @outside_compiled_ops_inside_tf_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK:            _else_func_name = "test_else_name"
    // CHECK-SAME:       _then_func_name = "test_then_name"
    // CHECK-SAME:       is_stateless = false
    // CHECK-NEXT:         %[[ARG_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK:              "tf.D"(%[[ARG_RECV_OUTPUT]]#0, %[[ARG_RECV_OUTPUT]]#1)
    // CHECK-NOT:          "tf._XlaSendFromHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:              "tf.Yield"() : () -> ()
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"
    // CHECK:            "tf._XlaHostComputeMlir"
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:            is_stateless = false
    // CHECK:              "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1", is_stateless = true} : (tensor<2xi32>, tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = true, _then_func_name = "test_then_name", _else_func_name = "test_else_name"} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op with dynamic shape.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if_dynamic_shape
  func.func @outside_compiled_ops_inside_tf_if_dynamic_shape(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:       %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:      %[[PREDICATE_RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:      key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[PREDICATE_RECV_OUTPUT]])
    // CHECK:            _else_func_name = "test_else_name"
    // CHECK-SAME        _then_func_name = "test_then_name"
    // CHECK-NEXT:         %[[ARG_RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK:              "tf.D"(%[[ARG_RECV_OUTPUT]]#0, %[[ARG_RECV_OUTPUT]]#1)
    // CHECK-NOT:          "tf._XlaSendFromHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:              "tf.Yield"() : () -> ()
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
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false, _then_func_name = "test_then_name", _else_func_name = "test_else_name"} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Ensures that separate send/recvs are added for values that are used by ops inside of multiple IfRegions.

  // CHECK-LABEL: func @outside_compiled_ops_multiple_tf_if
  func.func @outside_compiled_ops_multiple_tf_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%3) {_xla_outside_compilation = "auto"} : (tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        "tf.IfRegion"(%6) ({
          "tf.F"(%3) {_xla_outside_compilation = "auto"} : (tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        tf_device.return %3 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled

  // CHECK-LABEL: func @outside_compiled_tf_if
  func.func @outside_compiled_tf_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK-SAME:       (tensor<3x!tf_type.string>, tensor<i64>) -> (tensor<2xi32>, tensor<2xi32>, tensor<i1>)
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
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %7 = "tf.F"() : () -> tensor<2xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"(%4, %3, %7) {} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled and wrapped inside another
  // tf.IfRegion op

  // CHECK-LABEL: func @outside_compiled_tf_if_nested
  func.func @outside_compiled_tf_if_nested(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT_PREDICATE:[0-9]*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-SAME:       (tensor<3x!tf_type.string>, tensor<i64>) -> tensor<i1>
    // CHECK-NEXT:       tf.IfRegion"(%[[RECV_OUTPUT_PREDICATE]])
    // CHECK-NEXT:         %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:         key = "host_compute_channel_0_args"
    // CHECK-SAME:         (tensor<3x!tf_type.string>, tensor<i64>) -> (tensor<2xi32>, tensor<i1>)
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
    // CHECK:              "tf.Yield"() : () -> ()
    // CHECK:              "tf.Yield"() : () -> ()
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %7 = "tf.F"() : () -> tensor<2xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %8 = "tf.D"(%4, %3, %7) {} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)
          %9 = "tf.F"(%4) {} : (tensor<2xi32>) -> (tensor<i1>)

          "tf.IfRegion"(%9) ({
            "tf.H"(%8, %7) : (tensor<2xi32>, tensor<2xi32>) -> ()
            "tf.Yield"() : () -> ()
          }, {
            "tf.Yield"() : () -> ()
          }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion
  // op with return values.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if_with_return_values
  func.func @outside_compiled_ops_inside_tf_if_with_return_values(
    %arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    // CHECK:            "tf._XlaHostComputeMlir"(%[[G_OUTPUT]])
    // CHECK-SAME:       key = "if_predicate_channel_1"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK:              %[[HOST_COMPUTE_OUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK-SAME:         recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:         send_key = "host_compute_channel_0_args"
    // CHECK-NEXT:         "tf.Yield"(%[[HOST_COMPUTE_OUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %7 = "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)
          "tf.Yield"(%7) : (tensor<2xi32>) -> ()
        }, {

          %8 = "tf.F"() : () -> (tensor<2xi32>)
          "tf.Yield"(%8) : (tensor<2xi32>) -> ()
        }) { is_stateless = false} : (tensor<i1>) -> (tensor<2xi32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op without external inputs/outputs

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_if_without_input_outputs
  func.func @outside_compiled_ops_inside_tf_if_without_input_outputs(
    %arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    // CHECK:            "tf._XlaHostComputeMlir"(%[[G_OUTPUT]])
    // CHECK-SAME:       key = "if_predicate_channel_0"
    // CHECK-NEXT:       tf.IfRegion"(%[[G_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"() : () -> ()
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a nested
  // tf.IfRegion op.

  // CHECK-LABEL: func @outside_compiled_ops_inside_nested_if
  func.func @outside_compiled_ops_inside_nested_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
           %7 = "tf.H"(%4) : (tensor<2xi32>) -> (tensor<i1>)

          "tf.IfRegion"(%7)({
              "tf.Yield"() : () -> ()
            },
            {
              %8 = "tf.I"(%7) : (tensor<i1>) -> (tensor<2xi32>)
              "tf.D"(%8) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>) -> ()
              "tf.Yield"() : () -> ()
            }) { is_stateless = false} : (tensor<i1>) -> ()

          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) { is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a
  // tf.WhileRegion op body. Check that we mark the rewritten control-flow as
  // stateful even though the original control-flow is not (this is due to added
  // side-effecting communication ops).

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_body
  func.func @outside_compiled_ops_inside_tf_while_body(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       tf.WhileRegion"
    // CHECK:            is_stateless = false
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
    // CHECK:            is_stateless = false
    // CHECK:              %[[H_OUTPUT:[0-9]*]] = "tf.H"
    // CHECK-NEXT:         "tf.XlaSendToHost"(%[[H_OUTPUT]])
    // CHECK-NEXT:         "tf.Yield"(%[[H_OUTPUT]])
    // CHECK:              %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK-NEXT:         %[[HOST_COMPUTE_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"
    // CHECK-NEXT:         "tf.Yield"(%[[C_OUTPUT]], %[[HOST_COMPUTE_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.H"(%arg1) {is_stateless = true} :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %8 = "tf.C"(%arg1) {is_stateless = true} : (tensor<i32>) -> tensor<i32>
          %9 = "tf.D"(%arg1, %arg2) {_xla_outside_compilation = "cluster1", is_stateless = true} : (tensor<i32>, tensor<3xf32>) -> tensor<3xf32>
          "tf.Yield"(%8, %9) : (tensor<i32>, tensor<3xf32>) -> ()
        }) {is_stateless = true} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.WhileRegion op cond.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_cond
  func.func @outside_compiled_ops_inside_tf_while_cond(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.I"(%arg1, %arg2) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<3xf32>) -> tensor<i32>
          %8 = "tf.H"(%7) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%8) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %8 = "tf.D"(%arg1, %arg2) : (tensor<i32>, tensor<3xf32>) -> tensor<3xf32>
          "tf.Yield"(%7, %8) : (tensor<i32>, tensor<3xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.WhileRegion op cond and body.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_cond_body
  func.func @outside_compiled_ops_inside_tf_while_cond_body(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.I"(%arg1, %arg2) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<3xf32>) -> tensor<i32>
          %8 = "tf.H"(%7) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%8) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %8 = "tf.D"(%arg1, %arg2) {_xla_outside_compilation = "cluster2"} : (tensor<i32>, tensor<3xf32>) -> tensor<3xf32>
          "tf.Yield"(%7, %8) : (tensor<i32>, tensor<3xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of a single outside compiled cluster inside a tf.IfRegion op
  // nested in a tf.WhileRegion.

  // CHECK-LABEL: func @outside_compiled_ops_inside_tf_while_if
  func.func @outside_compiled_ops_inside_tf_while_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> tensor<3xf32>
            "tf.Yield"(%9) : (tensor<3xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<3xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<3xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<3xf32>) -> ()
        }) { is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of an outside compiled tf.IfRegion op where the entirety
  // of tf.IfRegion op is outside compiled with a nested tf.WhileRegion op.

  // CHECK-LABEL: func @outside_compiled_tf_if_nested_while
  func.func @outside_compiled_tf_if_nested_while(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:      %[[A_OUT:[0-9]*]] = "tf.A"
    // CHECK:      %[[F_OUT:[0-9]*]] = "tf.F"
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-NEXT:       %[[RECV_OUTPUT:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK-SAME:       (tensor<3x!tf_type.string>, tensor<i64>) -> (tensor<2xi32>, tensor<2xi32>, tensor<i1>)
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
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    %7 = "tf.F"() : () -> tensor<2xi32>

    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<2xi32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.IfRegion"(%6) ({
          %8 = "tf.D"(%4, %3, %7) {} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<3xf32>)
          %9 = "tf.J"() : () -> (tensor<i32>)
          %10 = "tf.K"() : () -> (tensor<i32>)
          "tf.WhileRegion"(%9, %8) ({
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
            %11 = "tf.I"(%arg1, %arg2) : (tensor<i32>, tensor<3xf32>) -> tensor<i32>
            %12 = "tf.H"(%10) :  (tensor<i32>) -> tensor<i1>
            "tf.Yield"(%12) : (tensor<i1>) -> ()
          }, {
          ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
            %11 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
            %12 = "tf.D"(%arg1, %arg2) : (tensor<i32>, tensor<3xf32>) -> tensor<3xf32>
            "tf.Yield"(%11, %12) : (tensor<i32>, tensor<3xf32>) -> ()
          }) { is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)
          "tf.Yield"() : () -> ()
        }, {
          "tf.Yield"() : () -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i1>) -> ()

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of an outside compiled tf.WhileRegion where the entire
  // tf.WhileRegion op is outside compiled with a nested tf.IfRegion.

  // CHECK-LABEL: func @outside_compiled_ops_tf_while_nested_if
  func.func @outside_compiled_ops_tf_while_nested_if(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)

        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"(%8) : (tensor<i32>) -> tensor<3xf32>
            "tf.Yield"(%9) : (tensor<3xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<3xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<3xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<3xf32>) -> ()
        }) {_xla_outside_compilation = "cluster1", is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Tests extraction of an outside compiled cluster that contains ops wrapped
  // inside multiple regions of nested tf.IfRegion and tf.WhileRegion.

  // CHECK-LABEL: func @outside_compiled_ops_with_multiple_region_single_cluster
  func.func @outside_compiled_ops_with_multiple_region_single_cluster(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>

    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:       %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
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
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<3xf32>)
        %4 = "tf.B"() {_xla_outside_compilation="cluster0"} : () -> (tensor<i32>)
        %6 = "tf.G"() : () -> (tensor<i1>)
        "tf.WhileRegion"(%4, %3) ({
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
          "tf.Yield"(%7) : (tensor<i1>) -> ()
        }, {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xf32>):
          %8 = "tf.C"(%4) {_xla_outside_compilation="cluster0"} : (tensor<i32>) -> tensor<i32>
          %10 = "tf.IfRegion"(%6) ({
            %9 = "tf.D"(%8) {_xla_outside_compilation="cluster0"} : (tensor<i32>) -> tensor<3xf32>
            "tf.Yield"(%9) : (tensor<3xf32>) -> ()
          }, {
            "tf.Yield"(%arg2) : (tensor<3xf32>) -> ()
          }) { is_stateless = false} : (tensor<i1>) -> tensor<3xf32>
          "tf.Yield"(%8, %10) : (tensor<i32>, tensor<3xf32>) -> ()
        }) {is_stateless = false} : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)

        %5 = "tf.E"() : () -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Verifies that ops in between outside compile ops and depending on results
  // from the host are moved after the host compute op so that dominance is not
  // violated. tf.C op in this case.
  // CHECK-LABEL: func @device_op_dominance
  func.func @device_op_dominance() -> () {
    // CHECK: tf.B
    // CHECK: tf._XlaSendFromHost
    // CHECK: tf._XlaRecvAtHost
    // CHECK: tf.D

    // CHECK: tf.A
    // CHECK: tf._XlaHostComputeMlir
    // CHECK: tf.C
    // CHECK: tf.E

    "tf_device.cluster"() ({
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"() {_xla_outside_compilation = "cluster0"} : () -> (tensor<i32>)
      "tf.C"(%1) : (tensor<i32>) -> ()
      "tf.D"(%1, %0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.E"(%0, %1) : (tensor<i32>, tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Verifies that ops indirectly depending on results from the host are also
  // moved after the host compute op. tf.E op in this case.

  // CHECK-LABEL: func @device_op_dominance_with_indirect_dependency
  func.func @device_op_dominance_with_indirect_dependency() -> () {
    // CHECK: tf.B
    // CHECK: tf._XlaRecvAtHost
    // CHECK: tf.F

    // CHECK: tf.A
    // CHECK: tf.C
    // CHECK: tf.D
    // CHECK: tf.E
    // CHECK: tf._XlaHostComputeMlir
    // CHECK: tf.G

    "tf_device.cluster"() ({
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"() {_xla_outside_compilation = "cluster0"} : () -> (tensor<i32>)
      %2 = "tf.C"(%1) : (tensor<i32>) -> (tensor<i32>)
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.E"(%2, %3) : (tensor<i32>, tensor<i32>) -> ()
      "tf.F"(%1, %0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.G"(%0, %1) : (tensor<i32>, tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests dynamically shaped input to outside compilation.

  // CHECK-LABEL: func @dynamically_shaped_input
  func.func @dynamically_shaped_input() -> () {
    // CHECK:          "tf_device.launch"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       host_mlir_module = ""
    "tf_device.cluster"() ({
      %0 = "tf.A"() : () -> (tensor<?xi32>)
      "tf.B"(%0) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests dynamically shaped output from outside compilation.

  // CHECK-LABEL: func @dynamically_shaped_output
  func.func @dynamically_shaped_output() -> () {
    // CHECK:          "tf_device.launch"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]]
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK-SAME:       host_mlir_module = "module  {\0A  func.func @host_func() -> tensor<?xi32> {\0A    %0 = \22tf.B\22() {_xla_outside_compilation = \22cluster1\22} : () -> tensor<?xi32> loc(#loc1)\0A    return %0 : tensor<?xi32> loc(#loc1)\0A  }
    // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
    "tf_device.cluster"() ({
      %0 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<?xi32>)
      "tf.C"(%0) : (tensor<?xi32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Tests extraction of a single outside compiled cluster with arg input and single dynamic shape device->host input.

  // CHECK-LABEL: func @mixed_dynamic_input_single_outside_compilation
  func.func @mixed_dynamic_input_single_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"(%[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK-SAME:       key = "host_compute_channel_0_args"
    // CHECK:            "tf.B"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%arg0, %[[A_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%arg0, %3) {_xla_outside_compilation = "cluster1"} : (tensor<2xi32>, tensor<?xi32>) -> (tensor<?xi32>)
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %5 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // CHECK-LABEL: func @dynamic_input_chained_ops_outside_compilation
  func.func @dynamic_input_chained_ops_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[B_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[C_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK:            "tf.D"(%[[HOST_OUTPUT]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %5 = "tf.C"(%4) {_xla_outside_compilation = "cluster1"}: (tensor<?xi32>) -> tensor<?xi32>
        %6 = "tf.D"(%5) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %6 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

   //  Tests that all inputs to a function are passed and all outputs are returned
   //  if any outputs of outside compilation cluster are dynamically shaped.

  // CHECK-LABEL: func @dynamic_input_all_captured_outside_compilation
  func.func @dynamic_input_all_captured_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[C_OUTPUT:[0-9]*]]:2 = "tf.C"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[C_OUTPUT]]#0, %[[C_OUTPUT]]#1, %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[E_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]]:2 = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[HOST_OUTPUT]]#0)
    // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[D_OUTPUT]], %[[HOST_OUTPUT]]#1)
    // CHECK:            "tf.F"(%[[HOST_OUTPUT2]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<?xi32>)
        %5:2 = "tf.C"(%3) {_xla_outside_compilation = "auto"} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<2xi32>)
        %6 = "tf.D"(%5#0) : (tensor<?xi32>) -> (tensor<?xi32>)
        %7 = "tf.E"(%6, %5#1) {_xla_outside_compilation = "auto"} : (tensor<?xi32>, tensor<2xi32>) -> (tensor<?xi32>)
        %8 = "tf.F"(%7) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %8 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  //  Tests that all inputs to a function are passed and all outputs are returned for an outside compilation cluster
  //  with dynamic-shaped inputs/outputs. This test case tests that if another outside compiled returns
  //  only statically-shaped outputs, the value consumed by an outside compiled op with some dynamically shaped inputs is
  //  sent back to the host.

  // CHECK-LABEL: func @dynamic_input_all_captured_mixed_shapes_outside_compilation
  func.func @dynamic_input_all_captured_mixed_shapes_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[C_OUTPUT:[0-9]*]]:2 = "tf.C"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[C_OUTPUT]]#0, %[[C_OUTPUT]]#1, %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[E_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]]:2 = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[HOST_OUTPUT]]#0)
    // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[HOST_OUTPUT]]#1)
    // CHECK:            "tf.F"(%[[HOST_OUTPUT2]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %5:2 = "tf.C"(%3) {_xla_outside_compilation = "auto"} : (tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
        %6 = "tf.D"(%5#0) : (tensor<2xi32>) -> (tensor<2xi32>)
        %7 = "tf.E"(%4, %5#1) {_xla_outside_compilation = "auto"} : (tensor<?xi32>, tensor<2xi32>) -> (tensor<?xi32>)
        %8 = "tf.F"(%7) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %8 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  //  Tests the case when an outside compiled op with only statically shaped inputs/outputs
  //  is in between two other outside compiled ops with some dynamic shapes.

  // CHECK-LABEL: func @static_shapes_sandwiched_outside_compilation
  func.func @static_shapes_sandwiched_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[C_OUTPUT:[0-9]*]]:2 = "tf.C"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[C_OUTPUT]]#0, %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[RECV_OUTPUT]])
    // CHECK:            "tf._XlaSendFromHostV2"(%[[E_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[F_OUTPUT:[0-9]*]] = "tf.F"(%[[C_OUTPUT]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[F_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[RECV_OUTPUT2:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[G_OUTPUT:[0-9]*]] = "tf.G"(%[[RECV_OUTPUT2]]#0, %[[RECV_OUTPUT2]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[G_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[HOST_OUTPUT]])
    // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[B_OUTPUT]])
    // CHECK:            %[[HOST_OUTPUT3:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK:            %[[HOST_OUTPUT4:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[HOST_OUTPUT3]], %[[HOST_OUTPUT2]])
    // CHECK:            "tf.H"(%[[HOST_OUTPUT4]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %5:2 = "tf.C"(%3) {_xla_outside_compilation = "auto"} : (tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
        %6 = "tf.D"(%5#0) : (tensor<2xi32>) -> (tensor<2xi32>)
        %7 = "tf.E"(%4) {_xla_outside_compilation = "auto"} : (tensor<?xi32>) -> (tensor<?xi32>)
        %8 = "tf.F"(%5#1) {_xla_outside_compilation = "auto"} : (tensor<2xi32>) -> (tensor<2xi32>)
        %9 = "tf.G"(%8, %7) {_xla_outside_compilation = "auto"} : (tensor<2xi32>, tensor<?xi32>) -> (tensor<?xi32>)
        %10 = "tf.H"(%9) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %10 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  //  Tests the case when an outside compiled op with some dynamically shaped input but static output
  //  comes before an outside compiled op with some dynamically shaped input/output.  This ensures that
  //  for the second outside compiled op, all of the inputs are explicitly provided.

  // CHECK-LABEL: func @static_output_dynamic_input_outside_compilation
  func.func @static_output_dynamic_input_outside_compilation(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %0 = "tf.A"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
    // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK-DAG:        %[[DEVICE_ORDINAL:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[C_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:            %[[RECV_OUTPUT2:[0-9]*]]:3 = "tf._XlaRecvAtHostV2"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[RECV_OUTPUT2]]#0, %[[RECV_OUTPUT2]]#1, %[[RECV_OUTPUT2]]#2)
    // CHECK:            "tf._XlaSendFromHostV2"(%[[E_OUTPUT]], %[[PROGRAM_OUTPUT]], %[[DEVICE_ORDINAL]])
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[HOST_OUTPUT]])
    // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._XlaHostComputeMlir"(%[[D_OUTPUT]], %[[B_OUTPUT]], %[[A_OUTPUT]])
    // CHECK:            "tf.F"(%[[HOST_OUTPUT2]])
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<2xi32>) {n = 2 : i32} {
      %2 = "tf_device.cluster"() ({
        %3 = "tf.A"() : () -> (tensor<2xi32>)
        %4 = "tf.B"() : () -> (tensor<?xi32>)
        %5 = "tf.C"(%4, %3) {_xla_outside_compilation = "auto"} : (tensor<?xi32>, tensor<2xi32>) -> (tensor<2xi32>)
        %6 = "tf.D"(%5) : (tensor<2xi32>) -> (tensor<2xi32>)
        %7 = "tf.E"(%6, %4, %3) {_xla_outside_compilation = "auto"} : (tensor<2xi32>, tensor<?xi32>, tensor<2xi32>) -> (tensor<?xi32>)
        %8 = "tf.F"(%7) : (tensor<?xi32>) -> tensor<2xi32>
        tf_device.return %8 : tensor<2xi32>
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }

    func.return %1 : tensor<2xi32>
  }

  // Check that a non-XLA value is not routed through the XLA side.

  // CHECK-LABEL: func @nonxla_static
  func.func @nonxla_static() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"(%0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>) -> tensor<!tf_type.string>
      %2 = "tf.C"(%1) {_xla_outside_compilation = "cluster0"} : (tensor<!tf_type.string>) -> tensor<i32>
      "tf.D"(%2) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Check that a non-XLA value with dynamic shape is not routed through the XLA side.

  // CHECK-LABEL: func @nonxla_dynamic
  func.func @nonxla_dynamic() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.A"() : () -> (tensor<i32>)
      %1 = "tf.B"(%0) {_xla_outside_compilation = "cluster0"} : (tensor<i32>) -> tensor<?x!tf_type.string>
      %2 = "tf.C"(%1) {_xla_outside_compilation = "cluster0"} : (tensor<?x!tf_type.string>) -> tensor<i32>
      "tf.D"(%2) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
    func.return
  }

  // Reproducer for an operand #x does not dominate this use

  // CHECK-LABEL: func @op_dominate_repro
  func.func @op_dominate_repro(%writer: tensor<*x!tf_type.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> () {
    %step = "tf.Const"() {device = "", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %tag = "tf.Const"() {device = "", value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %wmetadata = "tf.Const"() {device = "", value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %pred = "tf.Const"() {device = "", value = dense<0> : tensor<i1>} : () -> tensor<i1>
    "tf_device.cluster"() ({
      "tf.IfRegion"(%pred) ({
        %wtensor = "tf.Const"() {device = "", value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
        "tf.WriteSummary"(%writer, %step, %wtensor, %tag, %wmetadata) {_xla_outside_compilation = "auto"} : (tensor<*x!tf_type.resource>, tensor<i64>, tensor<f32>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()
        "tf.WriteSummary"(%writer, %step, %wtensor, %tag, %wmetadata) {_xla_outside_compilation = "auto"} : (tensor<*x!tf_type.resource>, tensor<i64>, tensor<f32>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()
        "tf.Yield"() : () -> ()
      }, {
        "tf.Yield"() : () -> ()
      }) {is_stateless = false} : (tensor<i1>) -> ()
      tf_device.return
    }) {_replication_info = "cluster__train_single_step", _xla_compile_device_type = "TPU", allow_soft_placement = true, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
    return
  }
}

// -----

// Tests that model parallelism does not affect outside compilation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  // CHECK-LABEL: func @outside_compilation_model_parallelism
  func.func @outside_compilation_model_parallelism() -> () {
     // CHECK: "tf_device.parallel_execute"
     // CHECK-NEXT: "tf_device.launch"
     // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:CPU:0"
     // CHECK-NEXT: "tf.B"
     // CHECK-NOT: _xla_outside_compilation
     // CHECK-NEXT:   tf_device.return
     // CHECK: "tf_device.cluster"
     // CHECK-NEXT: "tf.A"
     // CHECK: num_cores_per_replica = 2 : i64
    "tf_device.cluster"() ({
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
     }) {num_cores_per_replica = 2, topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1]} : () -> ()
    func.return
  }
}

// -----
// CHECK-LABEL: func @return_from_host_only
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1199 : i32}} {
  func.func @return_from_host_only() -> (tensor<?x300xi32>, tensor<?xi32>, tensor<?x4x!tf_type.string>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]]:3 = "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[RECV_OUTPUT]])
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[C_OUTPUT]])
    // CHECK:            %[[E_OUTPUT:[0-9]*]]:2 = "tf.E"(%[[D_OUTPUT]])
    // CHECK:            tf_device.return %[[D_OUTPUT:[0-9]*]], %[[E_OUTPUT:[0-9]*]]#0, %[[E_OUTPUT:[0-9]*]]#1
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-NEXT:       tf_device.return
    // CHECK:         return %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#1, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#2, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#0
    %0:3 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<300x?xi32>
      %2 = "tf.C"(%1) {_xla_outside_compilation = "auto0"} : (tensor<300x?xi32>) -> tensor<300x?x!tf_type.string>
      %3 = "tf.D"(%2) {_xla_outside_compilation = "auto1"} : (tensor<300x?x!tf_type.string>) -> tensor<?x4x!tf_type.string>
      %5:2 = "tf.E"(%3) {_xla_outside_compilation = "auto2"} : (tensor<?x4x!tf_type.string>) -> (tensor<?x300xi32>, tensor<?xi32>)
      tf_device.return %5#0, %5#1, %3 : tensor<?x300xi32>, tensor<?xi32>, tensor<?x4x!tf_type.string>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> (tensor<?x300xi32>, tensor<?xi32>, tensor<?x4x!tf_type.string>)
    return %0#0, %0#1, %0#2 : tensor<?x300xi32>, tensor<?xi32>, tensor<?x4x!tf_type.string>
  }
}

// -----
// CHECK-LABEL: func @return_from_host_and_tpu
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1199 : i32}} {
  func.func @return_from_host_and_tpu() -> (tensor<?xi32>, tensor<?x!tf_type.string>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            tf_device.return %[[B_OUTPUT:[0-9]*]]
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK-NEXT:       tf_device.return %[[C_OUTPUT:[0-9]*]]
    // CHECK:         return %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#1, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#0
    %0:2 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<?xi32>
      %2 = "tf.B"(%1) {_xla_outside_compilation = "auto0"} : (tensor<?xi32>) -> tensor<?x!tf_type.string>
      %3 = "tf.C"() : () -> tensor<?xi32>
      tf_device.return %3, %2 : tensor<?xi32>, tensor<?x!tf_type.string>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> (tensor<?xi32>, tensor<?x!tf_type.string>)
    return %0#0, %0#1 : tensor<?xi32>, tensor<?x!tf_type.string>
  }
}

// -----
// CHECK-LABEL: func @return_from_host_and_tpu_v2
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1199 : i32}} {
  func.func @return_from_host_and_tpu_v2() -> (tensor<?xi32>, tensor<?x!tf_type.string>, tensor<?x2xi32>, tensor<3x!tf_type.string>, tensor<?x3xi32>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[RECV_OUTPUT]])
    // CHECK:            tf_device.return %[[B_OUTPUT:[0-9]*]], %[[D_OUTPUT:[0-9]*]]
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[C_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_1_retvals"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"
    // CHECK-NEXT:       tf_device.return %[[A_OUTPUT:[0-9]*]], %[[C_OUTPUT:[0-9]*]], %[[E_OUTPUT:[0-9]*]]
    // CHECK:         return %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#2, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#0, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#3, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#1, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#4
    %0:5 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<?xi32>
      %2 = "tf.B"(%1) {_xla_outside_compilation = "auto0"} : (tensor<?xi32>) -> tensor<?x!tf_type.string>
      %3 = "tf.C"() : () -> tensor<?x2xi32>
      %4 = "tf.D"(%3) {_xla_outside_compilation = "auto1"} : (tensor<?x2xi32>) -> tensor<3x!tf_type.string>
      %5 = "tf.E"() : () -> tensor<?x3xi32>
      tf_device.return %1, %2, %3, %4, %5 : tensor<?xi32>, tensor<?x!tf_type.string>, tensor<?x2xi32>, tensor<3x!tf_type.string>, tensor<?x3xi32>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> (tensor<?xi32>, tensor<?x!tf_type.string>, tensor<?x2xi32>, tensor<3x!tf_type.string>, tensor<?x3xi32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<?xi32>, tensor<?x!tf_type.string>, tensor<?x2xi32>, tensor<3x!tf_type.string>, tensor<?x3xi32>
  }

  // CHECK-LABEL: func @deplicated_return_from_host_and_tpu
  func.func @deplicated_return_from_host_and_tpu() -> (tensor<?x3xi32>, tensor<?x3xi32>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:4 = "tf_device.parallel_execute"
    // CHECK-NEXT:     %[[LAUNCH_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-DAG:        %[[PROGRAM_OUTPUT:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
    // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
    // CHECK:            _xla_has_host_transfer = true
    // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[RECV_OUTPUT]])
    // CHECK:            tf_device.return %[[D_OUTPUT:[0-9]*]], %[[D_OUTPUT:[0-9]*]]
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[A_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
    // CHECK:            "tf._XlaHostComputeMlir"(%[[C_OUTPUT]])
    // CHECK-SAME:       recv_key = "host_compute_channel_1_retvals"
    // CHECK:            %[[E_OUTPUT:[0-9]*]] = "tf.E"
    // CHECK-NEXT:       tf_device.return %[[E_OUTPUT:[0-9]*]], %[[E_OUTPUT:[0-9]*]]
    // CHECK:         return %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#2, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#3, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#0, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#1
    %0:4 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<?xi32>
      %2 = "tf.B"(%1) {_xla_outside_compilation = "auto0"} : (tensor<?xi32>) -> tensor<?x!tf_type.string>
      %3 = "tf.C"() : () -> tensor<?x2xi32>
      %4 = "tf.D"(%3) {_xla_outside_compilation = "auto1"} : (tensor<?x2xi32>) -> tensor<3x!tf_type.string>
      %5 = "tf.E"() : () -> tensor<?x3xi32>
      tf_device.return %5, %5, %4, %4 : tensor<?x3xi32>, tensor<?x3xi32>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> (tensor<?x3xi32>, tensor<?x3xi32>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<?x3xi32>, tensor<?x3xi32>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>
  }
}
// -----

// Tests that a an error is reported when the pass results in a cluster output
// with a non-XLA type. The simplest way this can happen is if the inputing op
// is not marked for outside compilation. In general control, data, and side
// effect dependencies that are not marked for outside compilation can cause
// this.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // expected-error @+2 {{result with a non-XLA type}}
  func.func @non_XLA_result() {
    %cluster = "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<!tf_type.string>
      tf_device.return %a : tensor<!tf_type.string>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> tensor<!tf_type.string>
    func.return
  }
}

// -----

// Tests that a an error is reported when the pass results in a cluster output
// with a non-XLA type, specifically a resource type.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // expected-error @+2 {{result with a non-XLA type}}
  func.func @resource_result() {
    %cluster = "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<!tf_type.resource<tensor<f32>>>
      tf_device.return %a : tensor<!tf_type.resource<tensor<f32>>>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> tensor<!tf_type.resource<tensor<f32>>>
    func.return
  }
}

// -----

// Tests that an error is reported when an op with _xla_outside_compilation has
// an ancestor with _xla_outside_compilation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @outside_comp_ancestor() {
    "tf_device.cluster"() ({
      "tf.WhileRegion"() ({
      ^bb0():
        // expected-error @+1 {{has an ancestor marked for outside compilation}}
        %1 = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i1>
	"tf.Yield"(%1) : (tensor<i1>) -> ()
      }, {
      ^bb0():
        "tf.Yield"() : () -> ()
      }) {_xla_outside_compilation = "cluster1", is_stateless = true} : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }
}

// -----
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0"]} {
  // CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation_not_replicated_in_generic_pipeline
  func.func @single_outside_compiled_output_single_outside_compilation_not_replicated_in_generic_pipeline(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
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
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> (tensor<2xi32>)
      %2 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<2xi32>)
      %3 = "tf.C"(%2) : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %3 : tensor<2xi32>
    }) : () -> tensor<2xi32>

    func.return %0 : tensor<2xi32>
  }
}

// -----
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0"]} {
  // CHECK-LABEL: func @single_outside_compiled_output_device_type
  func.func @single_outside_compiled_output_device_type(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
    // CHECK-NEXT:     "tf_device.launch"
    // CHECK:            %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
    // CHECK-NOT:        "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
    // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
    // CHECK-SAME:       device_ordinal = 0
    // CHECK-SAME:       device_type = "CPU"
    // CHECK-SAME:       key = "host_compute_channel_0_retvals"
    // CHECK:          "tf_device.cluster"
    // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
    // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._XlaHostComputeMlir"()
    // CHECK-SAME:       recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:       send_key = "host_compute_channel_0_args"
    // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> (tensor<2xi32>)
      %2 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<2xi32>)
      %3 = "tf.C"(%2) : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %3 : tensor<2xi32>
    }) {_xla_compile_device_type = "CPU"} : () -> tensor<2xi32>

    func.return %0 : tensor<2xi32>
  }
}

// -----

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1443 : i32}} {
  // Tests map_outside_compilation when there is no replication.
  // The sharding is:
  //   type: OTHER
  //   tile_assignment_dimensions: 2
  //   tile_assignment_dimensions: 1
  //   tile_assignment_devices: 0
  //   tile_assignment_devices: 1
  // Serialized string:
  //   "\08\03\1A\02\02\01\22\02\00\01"

  // CHECK-LABEL: func @map_outside_compilation_not_replicated
  func.func @map_outside_compilation_not_replicated() -> () {
    // CHECK:       "tf_device.parallel_execute"
    // CHECK:         "tf_device.launch"
    // CHECK:           %[[PROGRAM0:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:           %[[RECV0:.+]] = "tf._XlaRecvAtHost"(%[[PROGRAM0]])
    // CHECK-SAME:        device_ordinal = 0
    // CHECK-SAME:        key = "host_compute_channel_0_args"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:           %[[B0:.+]] = "tf.OpB"(%[[RECV0]]) : (tensor<2x2xi64>) -> tensor<2x2xi64>
    // CHECK:           "tf._XlaSendFromHost"(%[[B0]], %[[PROGRAM0]])
    // CHECK-SAME:        device_ordinal = 0
    // CHECK-SAME:        key = "host_compute_channel_0_retvals"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:         }, {
    // CHECK:           %[[PROGRAM1:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:           %[[RECV1:.+]] = "tf._XlaRecvAtHost"(%[[PROGRAM1]])
    // CHECK-SAME:        device_ordinal = 1
    // CHECK-SAME:        key = "host_compute_channel_0_args"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:           %[[B1:.+]] = "tf.OpB"(%[[RECV1]]) : (tensor<2x2xi64>) -> tensor<2x2xi64>
    // CHECK:           "tf._XlaSendFromHost"(%[[B1]], %[[PROGRAM1]])
    // CHECK-SAME:        device_ordinal = 1
    // CHECK-SAME:        key = "host_compute_channel_0_retvals"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:         }, {
    // CHECK:           "tf_device.cluster"
    // CHECK:             %[[A:.+]] = "tf.OpA"
    // CHECK:             %[[A_SHARD:.+]] = "tf.XlaSpmdFullToShardShape"(%[[A]]) <{dim = -1 : i64, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []}> : (tensor<2x2xi64>) -> tensor<1x2xi64>
    // CHECK:             %[[B:.+]] = "tf._XlaHostComputeMlir"(%[[A_SHARD]])
    // CHECK-SAME:          manual_sharding = true
    // CHECK-SAME:          recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:          send_key = "host_compute_channel_0_args"
    // CHECK:             %[[B_FULL:.+]] = "tf.XlaSpmdShardToFullShape"(%[[B]]) <{dim = -1 : i64, full_shape = #tf_type.shape<2x2>, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []}> : (tensor<1x2xi64>) -> tensor<2x2xi64>
    // CHECK:             "tf.OpC"(%[[B_FULL]])
    "tf_device.cluster"() ({
      %0 = "tf.OpA"() {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01"} : () -> tensor<2x2xi64>
      %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<2x2xi64>) -> tensor<2x2xi64>
      "tf.OpC"(%1) : (tensor<2x2xi64>) -> ()
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }
}

// -----

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:2", "/job:localhost/replica:0/task:0/device:TPU:3", "/job:localhost/replica:0/task:0/device:TPU:4", "/job:localhost/replica:0/task:0/device:TPU:5", "/job:localhost/replica:0/task:0/device:TPU:6", "/job:localhost/replica:0/task:0/device:TPU:7", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1458 : i32}} {
  // Tests map_outside_compilation when there is replication.
  // The sharding is:
  //   type: OTHER
  //   tile_assignment_dimensions: 2
  //   tile_assignment_dimensions: 1
  //   tile_assignment_devices: 0
  //   tile_assignment_devices: 1
  // Serialized string:
  //   "\08\03\1A\02\02\01\22\02\00\01"

  // CHECK-LABEL: func @map_outside_compilation_replicated
  func.func @map_outside_compilation_replicated() -> () {
    // CHECK:     tf_device.replicate
    // CHECK:       "tf_device.parallel_execute"
    // CHECK:           %[[PROGRAM0:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:           %[[DEVICE0:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-SAME:        logical_core = 0
    // CHECK:           %[[RECV0:.+]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM0]], %[[DEVICE0]])
    // CHECK-SAME:        key = "host_compute_channel_0_args"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:           %[[B0:.+]] = "tf.OpB"(%[[RECV0]]) : (tensor<2x2xi64>) -> tensor<2x2xi64>
    // CHECK:           "tf._XlaSendFromHostV2"(%[[B0]], %[[PROGRAM0]], %[[DEVICE0]])
    // CHECK-SAME:        key = "host_compute_channel_0_retvals"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:         }, {
    // CHECK:           %[[PROGRAM1:.+]] = "tf._XlaCompileMlirPlaceholderProgramKey"
    // CHECK:           %[[DEVICE1:.+]] = "tf._TPUDeviceOrdinalPlaceholder"
    // CHECK-SAME:        logical_core = 1
    // CHECK:           %[[RECV1:.+]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM1]], %[[DEVICE1]])
    // CHECK-SAME:        key = "host_compute_channel_0_args"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:           %[[B1:.+]] = "tf.OpB"(%[[RECV1]]) : (tensor<2x2xi64>) -> tensor<2x2xi64>
    // CHECK:           "tf._XlaSendFromHostV2"(%[[B1]], %[[PROGRAM1]], %[[DEVICE1]])
    // CHECK-SAME:        key = "host_compute_channel_0_retvals"
    // CHECK-SAME:        _xla_has_host_transfer = true
    // CHECK:         }, {
    // CHECK:           "tf_device.cluster"
    // CHECK:             %[[A:.+]] = "tf.OpA"
    // CHECK:             %[[A_SHARD:.+]] = "tf.XlaSpmdFullToShardShape"(%[[A]]) <{dim = -1 : i64, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []}> : (tensor<2x2xi64>) -> tensor<1x2xi64>
    // CHECK:             %[[B:.+]] = "tf._XlaHostComputeMlir"(%[[A_SHARD]])
    // CHECK-SAME:          manual_sharding = true
    // CHECK-SAME:          recv_key = "host_compute_channel_0_retvals"
    // CHECK-SAME:          send_key = "host_compute_channel_0_args"
    // CHECK:             %[[B_FULL:.+]] = "tf.XlaSpmdShardToFullShape"(%[[B]]) <{dim = -1 : i64, full_shape = #tf_type.shape<2x2>, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []}> : (tensor<1x2xi64>) -> tensor<2x2xi64>
    // CHECK:             "tf.OpC"(%[[B_FULL]])
    tf_device.replicate() {n = 4 : i32} {
      "tf_device.cluster"() ({
        %0 = "tf.OpA"() {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01"} : () -> tensor<2x2xi64>
        %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<2x2xi64>) -> tensor<2x2xi64>
        "tf.OpC"(%1) : (tensor<2x2xi64>) -> ()
        tf_device.return
      }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
      tf_device.return
    }
    return
  }
}

// -----

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1443 : i32}} {
  // Test that map_outside_compilation works with 0 inputs to the
  // _XlaHostComputeMlir op. In this case, %arg0 is not input to the
  // generated _XlaHostComputeMlir.
  // CHECK-LABEL: func @map_outside_compilation_0_inputs
  func.func @map_outside_compilation_0_inputs(%arg0 : tensor<2xi64>) -> () {
    // CHECK:             "tf._XlaHostComputeMlir"()
    "tf_device.cluster"() ({
      %1 = "tf.OpA"(%arg0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<2xi64>) -> tensor<2xi64>
      "tf.OpB"(%1) : (tensor<2xi64>) -> ()
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }
}

// -----

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1443 : i32}} {

  // Test that map_outside_compilation's inputs are not unranked.
  func.func @map_outside_compilation_must_be_ranked() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.OpA"() : () -> tensor<*xi64>
      // expected-error @+1 {{must be ranked}}
      %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<*xi64>) -> tensor<*xi64>
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }

  // Test that map_outside_compilation's inputs have rank >= 1.
  func.func @map_outside_compilation_must_have_rank_gte_1() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.OpA"() : () -> tensor<i64>
      // expected-error @+1 {{must have rank at least one}}
      %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<i64>) -> tensor<i64>
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }

  // Test that map_outside_compilation's inputs shapes are divisible by num_cores_per_replica.
  func.func @map_outside_compilation_div_num_cores_per_replica() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.OpA"() : () -> tensor<3xi64>
      // expected-error @+1 {{divisible by num_cores_per_replica}}
      %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<3xi64>) -> tensor<3xi64>
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }

  // Test that map_outside_compilation's preceeding ops have an _XlaSharding attribute.
  func.func @map_outside_compilation_explicit_sharding() -> () {
    "tf_device.cluster"() ({
      %0 = "tf.OpA"() : () -> tensor<2xi64>
      // expected-error @+1 {{should have an explicit sharding}}
      %1 = "tf.OpB"(%0) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<2xi64>) -> tensor<2xi64>
      tf_device.return
    }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    return
  }

  // Test that map_outside_compilation inputs have the same sharding.
  func.func @map_outside_compilation_same_sharding() -> () {
    tf_device.replicate() {n = 4 : i32} {
      "tf_device.cluster"() ({
        %0 = "tf.OpA"() {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01"} : () -> tensor<2x2xi64>
        %1 = "tf.OpB"() {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\02"} : () -> tensor<2x2xi64>
        // expected-error @+1 {{should have the same sharding}}
        %2 = "tf.OpC"(%0, %1) {_xla_map_outside_compilation = "0", _xla_outside_compilation = "from_launch"} : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
        "tf.OpD"(%2) : (tensor<2x2xi64>) -> ()
        tf_device.return
      }) {_xla_compile_device_type = "TPU", computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, padding_map = [], topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01*\02\08\01", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
      tf_device.return
    }
    return
  }
}
