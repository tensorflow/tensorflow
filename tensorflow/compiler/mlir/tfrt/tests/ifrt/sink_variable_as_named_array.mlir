// RUN: tf-tfrt-opt -split-input-file -sink-variable-as-named-array %s | FileCheck %s

// -----
// Basic test: all variables tensors are for devices and sinked as named ifrt arrays
//
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:   [[HANDLE2:%.*]] = "tf.VarHandleOp"
// CHECK-NEXT:   [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"([[HANDLE2]])
// CHECK-SAME:       device_sharding_config_proto_text = "sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } device_ids: 0 device_ids: 1 "
// CHECK-SAME:       name = "__y"
// CHECK-SAME:       used_by_host = false 
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"([[KEY]], %arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [0 : i32]}>
// CHECK-SAME:    : (tensor<!tf_type.string>, tensor<1x3xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %result = "tf.IfrtCall"(%2, %arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = []}> { __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 device_assignment { replica_count: 1 computation_count: 2 computation_devices { replica_device_ids: 0 } computation_devices { replica_device_ids: 1 } } use_spmd_for_xla_partitioning: true "} : (tensor<3x1xf32>, tensor<1x3xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// Variable tensor for host can still be used.
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK:  "tf.VarHandleOp"
// CHECK-NOT:  [[VARIABLE:%.*]] = "tf.ReadVariableOp"
// CHECK-NEXT:  [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"
// CHECK-SAME:    used_by_host = true
// CHECK-NEXT:  "tf.MatMul"(%arg0, [[FUTURE]])
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"(%arg0, [[KEY]]) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [1 : i32]}>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %3 = "tf.MatMul"(%arg0, %2) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %result = "tf.IfrtCall"(%arg0, %2) <{program_id = 6515870160938153680 : i64, variable_arg_indices = []}> { __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 3 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { dim { size: 3 } dim { size: 1 } } kind: PARAMETER sharding { } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 device_assignment { replica_count: 1 computation_count: 2 computation_devices { replica_device_ids: 0 } computation_devices { replica_device_ids: 1 } } use_spmd_for_xla_partitioning: true "} : (tensor<1x3xf32>, tensor<3x1xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// Variable tensor is only for host
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK:  "tf.VarHandleOp"
// CHECK-NOT:  [[VARIABLE:%.*]] = "tf.ReadVariableOp"
// CHECK-NEXT:  [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"
// CHECK-SAME:    used_by_host = true
// CHECK-NEXT:  [[RES:%.*]] = "tf.MatMul"(%arg0, [[FUTURE]])
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %3 = "tf.MatMul"(%arg0, %2) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    return %3: tensor<1x1xf32>
  }
}
