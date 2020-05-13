// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-head-tail-outside-compilation | FileCheck %s --dump-input-on-failure

// Tests extraction of a outside compiled ops at head of TPU computation.

func @single_head_outside_compilation(%arg0 : tensor<i32>) -> () {
  // CHECK:      tf_device.launch
  // CHECK:        "tf.A"
  // CHECK-NEXT:   tf_device.return
  //
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.C"
  // CHECK-NEXT:   tf_device.return
  "tf_device.cluster"() ( {
    "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
    "tf.B"() : () -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @multiple_head_outside_compilation
func @multiple_head_outside_compilation(%arg0 : tensor<i32>) -> () {
  // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"()
  // CHECK:        %[[A_OUT:.*]] = "tf.A"
  // CHECK:        %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
  // CHECK:        "tf.C"
  // CHECK-NEXT:   tf_device.return %[[B_OUT]]
  //
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.D"(%[[LAUNCH_OUT]])
  // CHECK-NEXT:   tf_device.return
  "tf_device.cluster"() ( {
    %0 = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> (tensor<i32>)
    %1 = "tf.B"(%0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> (tensor<i32>)
    "tf.C"(%1, %arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> ()
    "tf.D"(%1) : (tensor<i32>) -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @test_do_not_outside_compiled_ops_in_middle
func @test_do_not_outside_compiled_ops_in_middle(%arg0 : tensor<i32>) -> () {
  // CHECK-NOT:  tf_device.launch
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.A"
  // CHECK-NEXT:   "tf.B"
  // CHECK-NEXT:   "tf.C"
  // CHECK-NEXT:   tf_device.return
  "tf_device.cluster"() ( {
    %0 = "tf.A"(%arg0) {} : (tensor<i32>) -> (tensor<i32>)
    %1 = "tf.B"(%0) {_xla_outside_compilation = "cluster1"}: (tensor<i32>) -> (tensor<i32>)
    "tf.C"(%1) : (tensor<i32>) -> ()
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @test_ops_with_tpu_operands_not_extracted
func @test_ops_with_tpu_operands_not_extracted(%arg0 : tensor<i32>) -> () {
  // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"()
  // CHECK:        %[[A_OUT:.*]] = "tf.A"
  // CHECK:        %[[D_OUT:.*]] = "tf.D"(%[[A_OUT]])
  // CHECK-NEXT:   tf_device.return %[[D_OUT]]
  //
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.B"
  // CHECK:        "tf.C"
  // CHECK:        "tf.E"
  // CHECK-NEXT:   tf_device.return
  "tf_device.cluster"() ( {
    %0 = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> (tensor<i32>)
    %1 = "tf.B"() {} : () -> (tensor<i32>)
    %2 = "tf.C"(%arg0, %1) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    %3 = "tf.D"(%0) {_xla_outside_compilation = "cluster1"}: (tensor<i32>) -> (tensor<i32>)
    %4 = "tf.E"(%3) {} : (tensor<i32>) -> (tensor<i32>)
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
