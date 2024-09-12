// RUN: tf-tfrt-opt -split-input-file -rewrite-cluster-to-ifrt-call %s | FileCheck %s
// TODO(b/316226111): the printer may not guarantee the same order of fields. Rewrite the checks to be less sensitive to proto serialization formats.
// -----
// Non-SPMD: one input and one output
//
// CHECK-LABEL: func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
// CHECK-NEXT:  "tf.IfrtCall"(%arg0)
// CHECK-SAME:       {program_id = [[PROGRAM_ID:.*]] : i64, variable_arg_indices = []}
// CHECK-SAME:       (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK:    return
//
// CHECK:  func.func @_ifrt_program__func(%arg0: tensor<1x3xf32>)
// CHECK-SAME: __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 3 } } kind: PARAMETER sharding { } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 1 "
// CHECK-SAME: device_assignment = []
// CHECK-SAME: tfrt_ifrt_serving.program_id = [[PROGRAM_ID]] : i64
// CHECK:      return

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1704 : i32}} {
func.func @serving_default(%arg0: tensor<1x3xf32>) -> (tensor<1x3xf32>) {
  %outputs  =  "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  %0 = "tf_device.cluster_func"(%arg0) {_producer_name = "UNKNOWN", func = @_func, input_sharding_configuration = [""], num_cores_per_replica = 1 : i64, device_assignment = [], topology = "", output_sharding_configuration = [""], step_marker_location = "STEP_MARK_AT_ENTRY", use_spmd_for_xla_partitioning = false, use_tpu = true} : (tensor<1x3xf32>) -> (tensor<1x3xf32>)
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: @_func
func.func private @_func(%arg0: tensor<1x3xf32>) -> (tensor<1x3xf32>) {
  return %arg0 : tensor<1x3xf32>
}
}


// -----
// SPMD: one input and no return
//
// CHECK-LABEL: func.func @serving_default(%arg0: tensor<1x3xf32>) {
// CHECK-NEXT:  "tf.IfrtCall"(%arg0)
// CHECK-SAME:       {program_id = [[PROGRAM_ID:.*]] : i64, variable_arg_indices = []}
// CHECK-SAME:       (tensor<1x3xf32>) -> ()
// CHECK:    return
//
// CHECK:  func.func @_ifrt_program__func(%arg0: tensor<1x3xf32>)
// CHECK-SAME: __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 3 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true "
// CHECK-SAME: device_assignment = [0, 0, 0, 0, 0, 0, 0, 1]
// CHECK-SAME: tfrt_ifrt_serving.program_id = [[PROGRAM_ID]] : i64
// CHECK:      return

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1704 : i32}} {
func.func @serving_default(%arg0: tensor<1x3xf32>) -> () {
  %outputs  =  "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  "tf_device.cluster_func"(%arg0) {_producer_name = "UNKNOWN", func = @_func, input_sharding_configuration = ["{devices=[2,1]0,1}"], num_cores_per_replica = 2 : i64, device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", output_sharding_configuration = [], step_marker_location = "STEP_MARK_AT_ENTRY", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<1x3xf32>) -> ()
  return
}

// CHECK-LABEL: @_func
func.func private @_func(%arg0: tensor<1x3xf32>) -> () {
  return
}
}

// -----
// Multiple ifrt calls and have two sharded arguments

// CHECK-LABEL: func.func @serving_default(%arg0: tensor<3x1xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:  %0 = "tf.IfrtCall"(%arg1, %arg0)
// CHECK-SAME:       {program_id = [[PROGRAM_ID:.*]] : i64, variable_arg_indices = []
// CHECK-SAME:       (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:    %2 = "tf.IfrtCall"(%1, %arg0)
// CHECK-SAME:       {program_id = [[PROGRAM_ID]] : i64, variable_arg_indices = []
// CHECK-SAME:       (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    %3 = "tf.add"(%0, %2) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:    return
//
// CHECK:  func.func @_ifrt_program__func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-SAME:      device_assignment = [0, 0, 0, 0, 0, 0, 0, 1]
// CHECK-SAME:      tfrt_ifrt_serving.program_id = [[PROGRAM_ID]] : i64
// CHECK-NEXT:     %0 = "tf.MatMul"(%arg0, %arg1)
// CHECK:          return

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1704 : i32}} {
func.func @serving_default(%arg0: tensor<3x1xf32>,  %arg1: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
  %outputs  =  "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  %outputs_0 = "tf_device.cluster_func"(%arg1, %arg0) {_producer_name = "UNKNOWN", func = @_func, input_sharding_configuration = ["{devices=[2,1]0,1}", ""], num_cores_per_replica = 2 : i64, device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", output_sharding_configuration = [""], step_marker_location = "STEP_MARK_AT_ENTRY", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %duplicate_arg =  "tf.Identity"(%arg1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %outputs_1 = "tf_device.cluster_func"(%duplicate_arg, %arg0) {_producer_name = "UNKNOWN", func = @_func, input_sharding_configuration = ["{devices=[2,1]0,1}", ""], num_cores_per_replica = 2 : i64, device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", output_sharding_configuration = [""], step_marker_location = "STEP_MARK_AT_ENTRY", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %outputs_2 = "tf.add"(%outputs_0, %outputs_1): (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  return %outputs_2 : tensor<1x1xf32>
}

// CHECK-LABEL: @_func
func.func private @_func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> (tensor<1x1xf32>) {
  %outputs_0 =  "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  return %outputs_0 : tensor<1x1xf32>
}
}


// -----
// Missing topology and device assignment attribute in spmd is ok

// CHECK-LABEL: func.func @serving_default(%arg0: tensor<3x1xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:  %0 = "tf.IfrtCall"(%arg1, %arg0)
// CHECK-SAME:       {program_id = [[PROGRAM_ID:.*]] : i64, variable_arg_indices = []
// CHECK-SAME:       (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK:    return
//
// CHECK:  func.func @_ifrt_program__func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-SAME:       device_assignment = []
// CHECK-SAME:      tfrt_ifrt_serving.program_id = [[PROGRAM_ID]] : i64
// CHECK-NEXT:     %0 = "tf.MatMul"(%arg0, %arg1)
// CHECK:          return

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1704 : i32}} {
func.func @serving_default(%arg0: tensor<3x1xf32>,  %arg1: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
  %outputs  =  "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  %outputs_0 = "tf_device.cluster_func"(%arg1, %arg0) {_producer_name = "UNKNOWN", func = @_func, input_sharding_configuration = ["{devices=[2,1]0,1}", ""], num_cores_per_replica = 2 : i64, output_sharding_configuration = [""], step_marker_location = "STEP_MARK_AT_ENTRY", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  return %outputs_0 : tensor<1x1xf32>
}

// CHECK-LABEL: @_func
func.func private @_func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> (tensor<1x1xf32>) {
  %outputs_0 =  "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  return %outputs_0 : tensor<1x1xf32>
}
}

