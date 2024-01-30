// RUN: tf-tfrt-opt -split-input-file -sink-variable-as-named-array %s | FileCheck %s

// -----
// Basic test: all variables tensors are for devices and sinked as named ifrt arrays
//
// CHECK-LABEL: func.func @restore_graph
// CHECK-NEXT:  [[FILE:%.*]] = "tf.Const"
// CHECK-NEXT:  [[SLICE:%.*]] = "tf.Const"
// CHECK-NEXT:  [[NAME:%.*]] = "tf.Const"
// CHECK-NEXT:  [[TENSOR:%.*]] = "tf.RestoreV2"([[FILE]], [[NAME]], [[SLICE]])
// CHECK-NEXT:  "tf.IfrtLoadVariable"([[TENSOR]]) <{device_sharding_config_proto_text = "sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } device_ids: 0 device_ids: 1 ", name = "__y"}> : (tensor<3x1xf32>) -> ()
// CHECK-NEXT:  return
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"(%arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [0 : i32], variable_names = ["__y"]}>
// CHECK-SAME:    : (tensor<1x3xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @restore_graph() {
    %cst = "tf.Const"() <{value = dense<"/variables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0)  : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
 }
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %result = "tf.IfrtCall"(%2, %arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [], variable_names = []}> { __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 device_assignment { replica_count: 1 computation_count: 2 computation_devices { replica_device_ids: 0 } computation_devices { replica_device_ids: 1 } } use_spmd_for_xla_partitioning: true "} : (tensor<3x1xf32>, tensor<1x3xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// Variable tensor for host can still be used.
//
// CHECK-LABEL: func.func @restore_graph
// CHECK:  [[TENSOR:%.*]] = "tf.RestoreV2"
// CHECK-NEXT:  "tf.VarHandleOp"
// CHECK-NEXT:  "tf.AssignVariableOp"
// CHECK-NEXT:  "tf.IfrtLoadVariable"([[TENSOR]]) <{device_sharding_config_proto_text = "sharding { } device_ids: 0 device_ids: 1 ", name = "__y"}> : (tensor<3x1xf32>) -> ()
// CHECK-NEXT:  return
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-LABEL:  "tf.VarHandleOp"
// CHECK-LABEL:  "tf.ReadVariableOp"
// CHECK-LABEL:  "tf.MatMul"
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"(%arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [1 : i32], variable_names = ["__y"]}>
// CHECK-SAME:    : (tensor<1x3xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @restore_graph() {
    %cst = "tf.Const"() <{value = dense<"/variables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
 }
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %3 = "tf.MatMul"(%arg0, %2) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %result = "tf.IfrtCall"(%arg0, %2) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [], variable_names = []}> { __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 3 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 device_assignment { replica_count: 1 computation_count: 2 computation_devices { replica_device_ids: 0 } computation_devices { replica_device_ids: 1 } } use_spmd_for_xla_partitioning: true "} : (tensor<1x3xf32>, tensor<3x1xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}
