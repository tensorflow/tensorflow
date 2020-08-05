// RUN: tf-opt %s -tf-mark-ops-for-outside-compilation | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: func @unsupported_op
func @unsupported_op() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.UnsupportedOp"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @op_string_result
func @op_string_result() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.Const"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK-SAME: tf.string
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Const"() {value = dense<"x"> : tensor<!tf.string>} : () -> tensor<!tf.string>
    %3 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}
// CHECK-LABEL: func @op_string_operand
func @op_string_operand(%arg0: tensor<!tf.string>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.StringToNumber"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK-SAME: tf.string
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.StringToNumber"(%arg0) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
    %3 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @op_string_operand_string_result
func @op_string_operand_string_result(%arg0: tensor<!tf.string>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK-SAME: tf.string
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Identity"(%arg0)  : (tensor<!tf.string>) -> tensor<!tf.string>
    %3 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// Test that a tf.IfRegion op with a captured string operand is marked for outside compilation.
// CHECK-LABEL: func @if_region_captured_string
func @if_region_captured_string(%arg0: tensor<i1>, %arg1: tensor<!tf.string>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK: "tf.StringToNumber"
    // CHECK: _xla_outside_compilation = "auto", is_stateless = true
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.IfRegion"(%arg0) ( {
      %3 = "tf.StringToNumber"(%arg1) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     },  {
      %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %5 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
    tf_device.return %5 : tensor<f32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that ops with string results/operands inside a tf.IfRegion branch are marked for outside compilation.

// CHECK-LABEL: func @if_region_string_op
func @if_region_string_op(%arg0: tensor<i1>, %arg1: tensor<?xi32>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.IfRegion"(%arg0) ( {
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     },  {
      // CHECK: "tf.Const"() {_xla_outside_compilation = "auto", value = dense<"1.0"> : tensor<!tf.string>}
      // CHECK-NEXT: "tf.StringToNumber"
      // CHECK-SAME: _xla_outside_compilation
      %4 = "tf.Const"() {value = dense<"1.0"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      %5 = "tf.StringToNumber"(%4) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
    // CHECK: {is_stateless
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
    tf_device.return %6: tensor<f32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that ops with string results/operands inside a nested tf.IfRegion branch are marked for outside compilation.

// CHECK-LABEL: func @nested_if_region_string_op
func @nested_if_region_string_op(%arg0: tensor<i1>, %arg1: tensor<?xi32>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.IfRegion"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.IfRegion"(%arg0) ( {
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      },  {
       // CHECK: "tf.Const"() {value = dense<true> : tensor<i1>}
       // CHECK-NOT: _xla_outside_compilation
       %4 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
       %5 = "tf.IfRegion"(%4)({
         // CHECK: "tf.Const"() {_xla_outside_compilation = "auto", value = dense<"1.0"> : tensor<!tf.string>}
         // CHECK-NEXT: "tf.StringToNumber"
         // CHECK-SAME: _xla_outside_compilation
         %6 = "tf.Const"() {value = dense<"1.0"> : tensor<!tf.string>} : () -> tensor<!tf.string>
         %7 = "tf.StringToNumber"(%6) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
         "tf.Yield"(%7) : (tensor<f32>) -> ()
       },  {
         // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
         // CHECK-NOT: _xla_outside_compilation
         %8 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
         "tf.Yield"(%8) : (tensor<f32>) -> ()
       // CHECK: {is_stateless
       }){is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
       "tf.Yield"(%5) : (tensor<f32>) -> ()
    // CHECK: {is_stateless
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
    tf_device.return %9: tensor<f32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}
