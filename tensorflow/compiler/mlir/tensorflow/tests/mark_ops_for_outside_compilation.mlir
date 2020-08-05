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


// Test that a tf.IfRegion op with a captured string operand is marked for outside compilation.
// CHECK-LABEL: func @if_region_captured_string
func @if_region_captured_string(%arg0: tensor<i1>, %arg1: tensor<!tf.string>) -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK: "tf.D"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: _xla_outside_compilation
    // CHECK-SAME: is_stateless = true
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.IfRegion"(%arg0) ( {
      %3 = "tf.D"(%arg1) : (tensor<!tf.string>) -> tensor<?xi32>
      "tf.Yield"(%3) : (tensor<?xi32>) -> ()
     },  {
      %4 = "tf.H"() : () -> tensor<?xi32>
      "tf.Yield"(%4) : (tensor<?xi32>) -> ()
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<?xi32>)
    %5 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// Test that op with a string results/operands inside a tf.IfRegion branch is marked for outside compilation.

// CHECK-LABEL: func @if_region_string_op
func @if_region_string_op(%arg0: tensor<i1>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.IfRegion"(%arg0)({
      // CHECK: "tf.D"
      // CHECK-NOT: _xla_outside_compilation
      %3 = "tf.D"(%arg1) : (tensor<?xi32>) -> tensor<?xi32>
      "tf.Yield"(%3) : (tensor<?xi32>) -> ()
    },  {
      // CHECK: "tf.F"
      // CHECK-SAME: _xla_outside_compilation
      %4 = "tf.F"() : () -> tensor<!tf.string>
      // CHECK: "tf.G"
      // CHECK-SAME: _xla_outside_compilation
      %5 = "tf.G"(%4) : (tensor<!tf.string>) -> tensor<?xi32>
      %6 = "tf.H"() : () -> tensor<?xi32>
      "tf.Yield"(%6) : (tensor<?xi32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> tensor<?xi32>
    // CHECK: "tf.C"
    // CHECK-NOT: _xla_outside_compilation
    %7 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %7 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// Test that op with a string results/operands inside a tf.IfRegion branch is marked for outside compilation.

// CHECK-LABEL: func @nested_if_region_string_op
func @nested_if_region_string_op(%arg0: tensor<i1>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.A"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.A"() : () -> tensor<?xi32>
    %2 = "tf.IfRegion"(%arg0)({
      // CHECK: "tf.D"
      // CHECK-NOT: _xla_outside_compilation
      %3 = "tf.D"(%arg1) : (tensor<?xi32>) -> tensor<?xi32>
      "tf.Yield"(%3) : (tensor<?xi32>) -> ()
    },  {
      %4 = "tf.E"() : () -> tensor<i1>
      %5 = "tf.IfRegion"(%4)({
        // CHECK: "tf.F"
        // CHECK-NOT: _xla_outside_compilation
        %6 = "tf.F"(%arg1) : (tensor<?xi32>) -> tensor<?xi32>
        "tf.Yield"(%6) : (tensor<?xi32>) -> ()
      },  {
        // CHECK: "tf.G"
        // CHECK-SAME: _xla_outside_compilation
        %7 = "tf.G"() : () -> tensor<!tf.string>
        // CHECK: "tf.H"
        // CHECK-SAME: _xla_outside_compilation
        %8 = "tf.H"(%7) : (tensor<!tf.string>) -> tensor<?xi32>
        %9 = "tf.I"() : () -> tensor<?xi32>
        "tf.Yield"(%9) : (tensor<?xi32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> tensor<?xi32>
      "tf.Yield"(%5) : (tensor<?xi32>) -> ()
    }) {is_stateless = true} : (tensor<i1>) -> tensor<?xi32>
    // CHECK: "tf.C"
    // CHECK-NOT: _xla_outside_compilation
    %10 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    tf_device.return %10 : tensor<?xi32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
