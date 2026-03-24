// RUN: mlir-hlo-opt --stablehlo-ext-sanitize-discardable-attributes --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: module @module_known_attr
// CHECK-SAME: mhlo.frontend_attributes
// CHECK-SAME: mhlo.input_output_alias
// CHECK-SAME: mhlo.is_dynamic
// CHECK-SAME: mhlo.is_same_data_across_replicas
// CHECK-SAME: mhlo.num_partitions
// CHECK-SAME: mhlo.num_replicas
// CHECK-SAME: mhlo.spmd_output_sharding
// CHECK-SAME: mhlo.spmd_parameters_shardings
// CHECK-SAME: mhlo.use_auto_spmd_partitioning
// CHECK-SAME: mhlo.xla_entry_computation_parameter_layouts
// CHECK-SAME: mhlo.xla_entry_computation_parameter_tiles
// CHECK-SAME: mhlo.xla_entry_computation_result_layout
// CHECK-SAME: mhlo.xla_entry_computation_result_tiles
module @module_known_attr attributes {
  mhlo.frontend_attributes = "",
  mhlo.input_output_alias = "",
  mhlo.is_dynamic = "",
  mhlo.is_same_data_across_replicas = "",
  mhlo.num_partitions = "",
  mhlo.num_replicas = "",
  mhlo.spmd_output_sharding = "",
  mhlo.spmd_parameters_shardings = [""],
  mhlo.use_auto_spmd_partitioning = "",
  mhlo.xla_entry_computation_parameter_layouts = "",
  mhlo.xla_entry_computation_parameter_tiles = "",
  mhlo.xla_entry_computation_result_layout = "",
  mhlo.xla_entry_computation_result_tiles = ""
} {
  func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: module @module_unknown_attr
// CHECK-NOT: xla_tpu_user_reserved_hbm_bytes
module @module_unknown_attr attributes {xla.xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
  func.func @func_unknown_attr(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: func @func_known_attr
// CHECK-SAME: execution_thread
// CHECK-SAME: mhlo.frontend_attributes
func.func @func_known_attr(%arg0: tensor<2x2xi32>)
    -> tensor<2x2xi32> attributes {
      execution_thread = "host",
      mhlo.frontend_attributes = ""
    } {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @func_unknown_attr
// CHECK-NOT: xla_tpu_user_reserved_hbm_bytes
func.func @func_unknown_attr(%arg0: tensor<2x2xi32>)
    -> tensor<2x2xi32> attributes {xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @func_arg_known_attr
// CHECK-SAME: jax.buffer_donor
// CHECK-SAME: mhlo.frontend_attributes
// CHECK-SAME: mhlo.layout_mode
// CHECK-SAME: mhlo.parameter_replication
// CHECK-SAME: mhlo.sharding
// CHECK-SAME: tf.aliasing_output
func.func @func_arg_known_attr(%arg0: tensor<2x2xi32> {
    jax.buffer_donor = true,
    mhlo.frontend_attributes = "",
    mhlo.layout_mode = "{1,0}",
    mhlo.parameter_replication = [true],
    mhlo.sharding = "",
    tf.aliasing_output = 0 : i32
  }) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @func_arg_unknown_attr
// CHECK-NOT: mhlo.unknown_attr
func.func @func_arg_unknown_attr(%arg0: tensor<2x2xi32> {mhlo.unknown_attr = ""}) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @func_result_unknown_attr
// CHECK-NOT: mhlo.unknown_attr
func.func @func_result_unknown_attr(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32> {mhlo.unknown_attr = ""}) {
  return %arg0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @op_known_attr
// CHECK-NEXT: stablehlo.add
// CHECK-SAME: layout = ""
// CHECK-SAME: mhlo.frontend_attributes
// CHECK-SAME: mhlo.literal
// CHECK-SAME: mhlo.original_value
// CHECK-SAME: mhlo.sharding
// CHECK-SAME: result_layout
// CHECK-SAME: source_layout
// CHECK-SAME: xla_shape
func.func @op_known_attr(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = stablehlo.add %arg0, %arg0 {
    layout = "",
    mhlo.frontend_attributes = "",
    mhlo.literal = "",
    mhlo.original_value = "",
    mhlo.sharding = "mhlo.sharding",
    result_layout = "",
    source_layout = "",
    xla_shape = ""
  } : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @op_unknown_attr
// CHECK-NOT: xla.unknown_attr
func.func @op_unknown_attr(%arg0: tensor<2x2xi32>)
    -> tensor<2x2xi32> attributes {xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
  %0 = stablehlo.add %arg0, %arg0 {xla.unknown_attr = ""} : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// expected-error@+1 {{SDY attribute encountered: "sdy.somthing". Run SDY export pass prior to sanitizing unregistered attributes.}}
module @sdy_module_attr_error attributes {sdy.somthing = ""} {
  func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

// expected-error@+1 {{SDY attribute encountered: "sdy.somthing". Run SDY export pass prior to sanitizing unregistered attributes.}}
func.func @sdy_func_attr_error(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> attributes {sdy.somthing = ""} {
  return %arg0 : tensor<2x2xi32>
}

// -----

// expected-error@+1 {{SDY attribute encountered: "sdy.somthing". Run SDY export pass prior to sanitizing unregistered attributes.}}
func.func @sdy_func_arg_attr_error(%arg0: tensor<2x2xi32> {sdy.somthing = ""}) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}


// -----

func.func @sdy_op_attr_error(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{SDY attribute encountered: "sdy.somthing". Run SDY export pass prior to sanitizing unregistered attributes.}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.somthing = ""} : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
