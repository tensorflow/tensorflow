// RUN: tf-opt %s -tf-mark-ops-for-outside-compilation | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: func @unsupported_op_missing_soft_placement_attribute
func @unsupported_op_missing_soft_placement_attribute() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.UnsupportedOp"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @unsupported_op_soft_placement_false
func @unsupported_op_soft_placement_false() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.UnsupportedOp"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {allow_soft_placement = false, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @assert_op_string_operand
func @assert_op_string_operand(%arg0: tensor<!tf.string>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Assert"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.UnsupportedOp"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %t = constant dense<true> : tensor<i1>
    "tf.Assert"(%t, %arg0) {summarize = 3} : (tensor<i1>, tensor<!tf.string>) -> ()
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @tf2xla_fallback_op
func @tf2xla_fallback_op() -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.UnsupportedOp"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.Sinh"
    // CHECK-NOT: _xla_outside_compilation
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %3 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    %4 = "tf.Sinh"(%2) : (tensor<f32>) -> tensor<f32>
    tf_device.return %4 : tensor<f32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @ignore_embedding_ops
func @ignore_embedding_ops() -> () {
  "tf_device.cluster"() ( {
    // CHECK: "tf.RecvTPUEmbeddingActivations"
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.SendTPUEmbeddingGradients"
    // CHECK-NOT: _xla_outside_compilation
    %2:2 = "tf.RecvTPUEmbeddingActivations"() {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D"} : () -> (tensor<2x2xf32>, tensor<4x4xf32>)
    "tf.SendTPUEmbeddingGradients"(%2#0, %2#1) {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (tensor<2x2xf32>, tensor<4x4xf32>) -> ()
    tf_device.return
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
  return
}

// CHECK-LABEL: func @ignore_stack_ops
func @ignore_stack_ops(%arg0: tensor<i32>) -> () {
  "tf_device.cluster"() ( {
    // CHECK: "tf.StackV2"
    // CHECK-NOT: _xla_outside_compilation
    %0 = "tf.StackV2"(%arg0) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
    tf_device.return
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
  return
}

// CHECK-LABEL: func @ignore_const_foldable_ops
func @ignore_const_foldable_ops(%arg0: tensor<i32>) -> () {
  "tf_device.cluster"() ( {
    %s0 = "tf.Const"() {value = dense<[501, 1, 32, 1280]> : tensor<4xi32>} : () -> tensor<4xi32>
    %s1 = "tf.Const"() {value = dense<[  1, 1,  1, 1280]> : tensor<4xi32>} : () -> tensor<4xi32>

    // CHECK: "tf.BroadcastGradientArgs"
    // CHECK-NOT: _xla_outside_compilation
    %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<3xi32>)
    tf_device.return
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
  return
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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// Test that operations inside tf.IfRegion op are corrected marked for outside
// compilation.

// CHECK-LABEL: func @ops_inside_tf_if_outside_compiled
func @ops_inside_tf_if_outside_compiled(%arg0: tensor<i1>, %arg1: tensor<!tf.string>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK:      "tf.Const"() {value = dense<1> : tensor<i32>}
    // CHECK-NOT:  _xla_outside_compilation
    // CHECK:      "tf.IfRegion"
    // CHECK:        "tf.StringToNumber"
    // CHECK-SAME:   _xla_outside_compilation
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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that ops with string results/operands inside a tf.IfRegion branch are
// marked for outside compilation.

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
      // CHECK: "tf.Const"() {_xla_outside_compilation = "auto0", value = dense<"1.0"> : tensor<!tf.string>}
      // CHECK-NEXT: "tf.StringToNumber"
      // CHECK-SAME: _xla_outside_compilation
      %4 = "tf.Const"() {value = dense<"1.0"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      %5 = "tf.StringToNumber"(%4) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
    // CHECK: {is_stateless
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
    tf_device.return %6: tensor<f32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that ops with string results/operands inside a nested tf.IfRegion branch
// are marked for outside compilation.

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
         // CHECK: "tf.Const"() {_xla_outside_compilation = "auto0", value = dense<"1.0"> : tensor<!tf.string>}
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
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that ops inside tf.WhileRegion op are correct marked for outside
// compilation.

// CHECK-LABEL: func @ops_inside_while_outside_compiled
func @ops_inside_while_outside_compiled(%arg0: tensor<i32>, %arg1: tensor<!tf.string>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK:     "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK:     "tf.WhileRegion"
    // CHECK:       "tf.StringToNumber"
    // CHECK-SAME:   _xla_outside_compilation
    %1 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %2:2 = "tf.WhileRegion"(%1, %arg0) ( {
      ^bb0(%carg0: tensor<f32>, %carg1: tensor<i32>):
         %limit = constant dense<5> : tensor<i32>
         %cond = "tf.NotEqual"(%carg1, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
         "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },  {
      ^bb0(%barg0: tensor<f32>, %barg1: tensor<i32>):
        %one = constant dense<1> : tensor<i32>
        %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %3 = "tf.StringToNumber"(%arg1) {out_type = f32} : (tensor<!tf.string>) -> tensor<f32>
        "tf.Yield"(%3, %sub) : (tensor<f32>, tensor<i32>) -> ()
    }) {is_stateless = true} : (tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %5 = "tf.Identity"(%2#0) : (tensor<f32>) -> (tensor<f32>)
    tf_device.return %5 : tensor<f32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Test that an unsupported op within a  tf.WhileRegion is marked for outside compilation.

// CHECK-LABEL: func @while_region_unsupported_op
func @while_region_unsupported_op(%arg0: tensor<i32>, %arg1: tensor<!tf.string>) -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    // CHECK-NOT: _xla_outside_compilation
    // CHECK: "tf.WhileRegion"
    %1 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %2:2 = "tf.WhileRegion"(%1, %arg0) ( {
      ^bb0(%carg0: tensor<f32>, %carg1: tensor<i32>):
         %limit = constant dense<5> : tensor<i32>
         %cond = "tf.NotEqual"(%carg1, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
         "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },  {
      ^bb0(%barg0: tensor<f32>, %barg1: tensor<i32>):
        %one = constant dense<1> : tensor<i32>
        %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        // CHECK: "tf.UnsupportedOp"
        // CHECK-SAME: _xla_outside_compilation
        %3 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
        // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
        %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
        "tf.Yield"(%4, %sub) : (tensor<f32>, tensor<i32>) -> ()
    // CHECK: {is_stateless = true
    }) {is_stateless = true} : (tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
    // CHECK: "tf.Identity"
    // CHECK-NOT: _xla_outside_compilation
    %5 = "tf.Identity"(%2#0) : (tensor<f32>) -> (tensor<f32>)
    tf_device.return %5 : tensor<f32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// Checks that ops with inputs and outputs with string subtypes are marked
// for outside compilation.

// CHECK-LABEL: func @check_op_with_variant_string_subtypes_outside_compiled
func @check_op_with_variant_string_subtypes_outside_compiled(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> () {
  "tf_device.cluster"() ( {
    // CHECK:      "tf.TensorListReserve"
    // CHECK-SAME: _xla_outside_compilation
    // CHECK:      "tf.TensorListGetItem"
    // CHECK-SAME: _xla_outside_compilation
    %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf.variant<tensor<*x!tf.string>>>
    "tf.TensorListGetItem"(%0, %arg1, %arg2) : (tensor<!tf.variant<tensor<*x!tf.string>>>, tensor<i32>, tensor<3xi32>) -> tensor<24x24x64xui8>
    tf_device.return
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
  return
}
// CHECK-LABEL: func @check_op_with_resource_string_subtypes_outside_compiled
func @check_op_with_resource_string_subtypes_outside_compiled(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf.resource<tensor<!tf.string>>>) -> () {
  "tf_device.cluster"() ( {
    // CHECK:      "tf.VarHandleOp"
    // CHECK-SAME: _xla_outside_compilation
    "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "", shared_name = ""} : () -> tensor<!tf.resource<tensor<!tf.string>>>
    tf_device.return
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> ()
  return
}

