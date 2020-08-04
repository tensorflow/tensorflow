// RUN: tf-opt %s -tf-mark-ops-for-outside-compilation | FILECHECK_OPTS="" FileCheck %s


// CHECK-LABEL: func @op_string_result
func @op_string_result() -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.B"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.C"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.B"(%1) : (tensor<?xi32>) -> tensor<!tf.string>
    %3 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %3 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @op_string_operand
func @op_string_operand(%arg0: tensor<!tf.string>) -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.B"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.C"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.B"(%arg0) : (tensor<!tf.string>) -> tensor<?xi32>
    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %3 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @op_string_operand_string_result
func @op_string_operand_string_result(%arg0: tensor<!tf.string>) -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.B"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.C"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.B"(%arg0) : (tensor<!tf.string>) -> tensor<!tf.string>
    %3 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %3 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
