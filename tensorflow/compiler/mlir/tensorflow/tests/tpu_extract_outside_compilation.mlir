// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-outside-compilation | FileCheck %s --dump-input-on-failure

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
  // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK: %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
      // CHECK-NEXT: "tf_device.launch"
      // CHECK: %[[TPU_CLUSTER_OUTPUT:[0-9]*]] = "tf_device.cluster"
        // CHECK: tf_device.return
      // CHECK: tf_device.return %[[TPU_CLUSTER_OUTPUT]]
      // CHECK: tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
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
  // CHECK: %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
    // CHECK: %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2  = "tf_device.parallel_execute"
      // CHECK-NEXT: "tf_device.launch"
      // CHECK: %[[TPU_CLUSTER_OUTPUT:[0-9]*]]:2 = "tf_device.cluster"
        // CHECK: tf_device.return
      // CHECK: tf_device.return %[[TPU_CLUSTER_OUTPUT]]
    // CHECK: tf_device.return %[[PARALLEL_EXECUTE_OUTPUT]]
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

// TODO(b/154363171): Add test cases for when output of outside compilation is returned by parallel_execute.
