// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-outside-compilation | FileCheck %s --dump-input-on-failure

// Tests that missing `_xla_outside_compilation` attribute value results in an error.

func @missing_outside_compilation_attribute() -> () {
  "tf_device.launch"() ( {
    "tf.A"() : () -> ()
    // expected-error@+1 {{attribute '_xla_outside_compilation' is empty}}
    "tf.B"() {_xla_outside_compilation = ""} : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// -----

// Tests that TPU cluster with no outside compilation does not generate parallel_execute.

// CHECK-LABEL: func @no_outside_compilation
func @no_outside_compilation() -> tensor<?xi32> {
  %0 = "tf_device.launch"() ( {
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.B"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %2 : tensor<?xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<?xi32>
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
   // CHECK: "tf_device.launch"
   // CHECK-NEXT: "tf.A"
   // CHECK: device = "tpu0"
   // CHECK-SAME: launch_attr = "launch_attr"
  "tf_device.launch"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
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
   // CHECK: "tf_device.launch"
   // CHECK-NEXT: "tf.A"
   // CHECK-NEXT: "tf.E"
   // CHECK: device = "tpu0"
   // CHECK-SAME: launch_attr = "launch_attr"
  "tf_device.launch"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.D"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.E"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// Tests extraction of a multiple outside compiled clusters with no input or output dependecies.

// CHECK-LABEL: func @nodep_multiple_outside_compilation
func @nodep_multiple_outside_compilation() -> () {
   // CHECK: "tf_device.parallel_execute"
   // CHECK-COUNT-3: "tf_device.launch"
  "tf_device.launch"() ( {
    "tf.A"() : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() : () -> ()
    "tf.D"() {_xla_outside_compilation = "cluster2"} : () -> ()
    "tf.E"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
