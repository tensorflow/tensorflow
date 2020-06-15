// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-outside-compilation | FileCheck %s

// Tests that missing `_xla_outside_compilation` attribute value results in an error.

func @missing_outside_compilation_attribute() -> () {
  "tf_device.cluster"() ( {
    "tf.A"() : () -> ()
    // expected-error@+1 {{attribute '_xla_outside_compilation' is empty}}
    "tf.B"() {_xla_outside_compilation = ""} : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// -----

// Tests that TPU cluster with no outside compilation does not generate parallel_execute.

// CHECK-LABEL: func @no_outside_compilation
func @no_outside_compilation() -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.B"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %2 : tensor<?xi32>
  }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-NOT: "tf_device.parallel_execute"

// Tests extraction of a single outside compiled cluster with no input or output dependecies.

// CHECK-LABEL: func @nodep_single_outside_compilation
func @nodep_single_outside_compilation() -> () {
   // CHECK: "tf_device.parallel_execute"
   // CHECK-NEXT: "tf_device.launch"
   // CHECK-NEXT: "tf.B"
   // CHECK-NOT: _xla_outside_compilation
   // CHECK: "tf_device.cluster"
   // CHECK-NEXT: "tf.A"
   // CHECK: cluster_attr = "cluster_attr"
  "tf_device.cluster"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
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
   // CHECK: cluster_attr = "cluster_attr"
  "tf_device.cluster"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.E"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Tests extraction of a multiple outside compiled clusters with no input or output dependecies.

// CHECK-LABEL: func @nodep_multiple_outside_compilation
func @nodep_multiple_outside_compilation() -> () {
   // CHECK: "tf_device.parallel_execute"
   // CHECK-COUNT-2: "tf_device.launch"
   // CHECK: "tf_device.cluster"
  "tf_device.cluster"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() : () -> ()
    "tf.D"() {_xla_outside_compilation = "cluster2"} : () -> ()
    "tf.E"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Tests extraction of a single outside compiled cluster with single TPU cluster return.

// CHECK-LABEL: func @single_tpu_return_single_outside_compilation
func @single_tpu_return_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
  // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
  // CHECK-NEXT:     "tf_device.launch"
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
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
    }) {cluster_attr = "cluster_attr"} : () -> (tensor<?xf32>, tensor<?xi32>)
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            "tf.B"(%[[RECV_OUTPUT]])
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
      %4 = "tf.C"() : () -> tensor<?xi32>
      tf_device.return %4 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
    tf_device.return %2 : tensor<?xi32>
  }

  return %1 : tensor<?xi32>
}

// Tests extraction of a single outside compiled cluster with single host->device output.

// CHECK-LABEL: func @single_outside_compiled_output_single_outside_compilation
func @single_outside_compiled_output_single_outside_compilation(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  // CHECK:      %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
  // CHECK:        %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
  // CHECK-NEXT:     "tf_device.launch"
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"()
  // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:         "tf_device.cluster"
  // CHECK:           %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:           %[[HOST_OUTPUT:[0-9]*]] = "tf._HostComputeMlir"()
  // CHECK-SAME:      key = "host_compute_channel_cluster1"
  // CHECK:           "tf.C"(%[[HOST_OUTPUT]])
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %4 = "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> (tensor<?xi32>)
      %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %5 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
  // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            tf_device.return %[[HOST_OUTPUT]]
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
      %5 = "tf.C"(%3) : (tensor<?xi32>) -> (tensor<?xi32>)
      tf_device.return %4 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT]])
  // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            %[[HOST_OUTPUT:[0-9]*]] = "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            "tf.C"(%[[HOST_OUTPUT]])
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
      %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %5 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK:            %[[B_OUTPUT:[0-9]*]]:2 = "tf.C"(%[[RECV_OUTPUT]]#0, %[[RECV_OUTPUT]]#1)
  // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]]#0, %[[B_OUTPUT]]#1, %[[PROGRAM_OUTPUT]])
  // CHECK-SAME:        key = "host_compute_channel_cluster1"
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK:            %[[HOST_OUTPUT:[0-9]*]]:2 = "tf._HostComputeMlir"(%[[A_OUTPUT]], %[[B_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
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
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT2:[a-z_0-9]*]], %[[PROGRAM_OUTPUT2:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT2:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT2]])
  // CHECK:            %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[RECV_OUTPUT2]])
  // CHECK:            "tf._XlaSendFromHost"(%[[D_OUTPUT]], %[[PROGRAM_OUTPUT]])
  // CHECK-SAME:         key = "host_compute_channel_cluster2"
  // CHECK:          "tf_device.launch"
  // CHECK:            %[[STATUS_OUTPUT1:[a-z_0-9]*]], %[[PROGRAM_OUTPUT1:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT1:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT1]])
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[RECV_OUTPUT1]])
  // CHECK:            "tf._XlaSendFromHost"(%[[B_OUTPUT]], %[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            %[[HOST_OUTPUT1:[0-9]*]] = "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[HOST_OUTPUT1]])
  // CHECK:            %[[HOST_OUTPUT2:[0-9]*]] = "tf._HostComputeMlir"(%[[C_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster2"
  // CHECK:            "tf.E"(%[[HOST_OUTPUT2]])
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %4 = "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> (tensor<?xi32>)
      %5 = "tf.C"(%4) : (tensor<?xi32>) -> (tensor<?xi32>)
      %6 = "tf.D"(%5) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> (tensor<?xi32>)
      %7 = "tf.E"(%6) : (tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %7 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            "tf.B"(%arg0, %[[RECV_OUTPUT]])
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      "tf.B"(%arg0, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> ()
      %4 = "tf.C"() : () -> tensor<?xi32>
      tf_device.return %4 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT_2:[a-z_0-9]*]], %[[PROGRAM_OUTPUT_2:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT_2:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT_2]])
  // CHECK-SAME:      key = "host_compute_channel_cluster2"
  // CHECK:           "tf.D"(%[[RECV_OUTPUT_2]])
  // CHECK:          "tf_device.launch"
  // CHECK:            %[[STATUS_OUTPUT_1:[a-z_0-9]*]], %[[PROGRAM_OUTPUT_1:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT_1:[0-9]*]] = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT_1]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            "tf.B"(%[[RECV_OUTPUT_1]])
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            "tf._HostComputeMlir"(%[[A_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            %[[C_OUTPUT:[0-9]*]] = "tf.C"
  // CHECK:            "tf._HostComputeMlir"(%[[C_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster2"
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      "tf.B"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
      %4 = "tf.C"() : () -> tensor<?xi32>
      "tf.D"(%4) {_xla_outside_compilation = "cluster2"} : (tensor<?xi32>) -> ()
      tf_device.return %4 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
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
  // CHECK:            %[[STATUS_OUTPUT:[a-z_0-9]*]], %[[PROGRAM_OUTPUT:[a-z_0-9]*]] = "tf._TPUCompileMlir"
  // CHECK:            %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHost"(%[[PROGRAM_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  // CHECK:            "tf.C"(%[[RECV_OUTPUT]]#0)
  // CHECK:            "tf.D"(%[[RECV_OUTPUT]]#1, %[[RECV_OUTPUT]]#0)
  // CHECK:          "tf_device.cluster"
  // CHECK:            %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK:            %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK:            "tf._HostComputeMlir"(%[[A_OUTPUT]], %[[B_OUTPUT]])
  // CHECK-SAME:       key = "host_compute_channel_cluster1"
  %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
    %2 = "tf_device.cluster"() ( {
      %3 = "tf.A"() : () -> (tensor<?xi32>)
      %4 = "tf.B"() : () -> (tensor<?xi32>)
      "tf.C"(%3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
      "tf.D"(%4, %3) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>, tensor<?xi32>) -> ()
      %5 = "tf.E"() : () -> tensor<?xi32>
      tf_device.return %5 : tensor<?xi32>
    }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
    tf_device.return %2 : tensor<?xi32>
  }

  return %1 : tensor<?xi32>
}
