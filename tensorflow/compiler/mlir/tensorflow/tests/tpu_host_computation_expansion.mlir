// RUN: tf-opt %s -split-input-file -tf-tpu-host-computation-expansion | FileCheck %s

// Tests expansion of a outside compiled ops at head/tail of TPU computation.

// CHECK-LABEL: func @identity_at_head_expanded
func @identity_at_head_expanded(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-SAME: _xla_outside_compilation = ""
  "tf_device.cluster"() ( {
    %1 = "tf.Identity"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.B"(%1) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) : () -> ()
  return
}

// CHECK-LABEL: func @cast_at_head_expanded
func @cast_at_head_expanded(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Cast"
  // CHECK-SAME: _xla_outside_compilation = ""
  "tf_device.cluster"() ( {
    %1 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.B"(%1) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {} : () -> ()
  return
}

// CHECK-LABEL: func @check_consecutive_unary_ops_outside_compiled
func @check_consecutive_unary_ops_outside_compiled(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Cast"
  // CHECK-SAME: _xla_outside_compilation = ""
  // CHECK-NEXT: "tf.Identity"
  // CHECK-SAME: _xla_outside_compilation = ""
  // CHECK-NEXT: "tf.B"
  "tf_device.cluster"() ( {
    %1 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    %2 = "tf.Identity"(%1) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.B"(%2) {_xla_outside_compilation = "cluster1"} : (tensor<?xi32>) -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {} : () -> ()
  return
}

// CHECK-LABEL: func @check_only_necesarily_ops_outside_compiled
func @check_only_necesarily_ops_outside_compiled(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NOT: _xla_outside_compilation = ""
  // CHECK-NEXT: "tf.B"
  "tf_device.cluster"() ( {
    %1 = "tf.Identity"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.B"(%1) : (tensor<?xi32>) -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) {} : () -> ()
  return
}

// CHECK-LABEL: func @check_only_necesarily_ops_outside_compiled_with_chained_ops
func @check_only_necesarily_ops_outside_compiled_with_chained_ops(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Cast"
  // CHECK-NOT: _xla_outside_compilation
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NOT: _xla_outside_compilation
  // CHECK-NEXT: "tf.B"
  "tf_device.cluster"() ( {
    %1 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    %2 = "tf.Identity"(%1) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.B"(%2) : (tensor<?xi32>) -> ()
    "tf.C"() : () -> ()
    tf_device.return
  }) : () -> ()
  return
}

// CHECK-LABEL: func @check_op_without_usage_not_outside_compiled
func @check_op_without_usage_not_outside_compiled(%arg0: tensor<?xi32>) {
  // CHECK: "tf_device.cluster"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NOT: _xla_outside_compilation
  "tf_device.cluster"() ( {
    "tf.Identity"(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    "tf.C"() : () -> ()
    tf_device.return
  }) : () -> ()
  return
}
