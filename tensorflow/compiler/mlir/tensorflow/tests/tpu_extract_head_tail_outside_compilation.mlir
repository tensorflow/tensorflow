// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-head-tail-outside-compilation | FileCheck %s --dump-input-on-failure

// Tests extraction of a single outside compiled cluster with no input or output dependecies.

// CHECK-LABEL: func @nodep_single_head_outside_compilation
func @nodep_single_head_outside_compilation() -> () {
   // CHECK: "tf.A"
   // CHECK-NEXT: "tf_device.launch"
  "tf_device.launch"() ( {
    "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.B"() : () -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @nodep_multiple_head_outside_compilation
func @nodep_multiple_head_outside_compilation() -> () {
   // CHECK: "tf.A"
   // CHECK-NEXT: "tf.B"
   // CHECK-NEXT: "tf_device.launch"
  "tf_device.launch"() ( {
    "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
