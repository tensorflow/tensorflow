// RUN: tf-opt %s -tf-replicated-clustering-bridge-v2 --mlir-print-ir-before-all --mlir-print-ir-after-all | FileCheck %s

// CHECK-LABEL:func.func @main
// CHECK %0 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster__train_helper", device = ""} : () -> tensor<!tf_type.string>
// CHECK %1 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> tensor<25x5xf32>
// CHECK %2 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> tensor<25x5xf32>
// CHECK %cst = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %cst_0 = "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %3 = "tf.Fill"(%cst_0, %cst) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %cst_1 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %cst_2 = "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %4 = "tf.Fill"(%cst_2, %cst_1) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %cst_3 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %cst_4 = "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %5 = "tf.Fill"(%cst_4, %cst_3) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %cst_5 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %cst_6 = "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %6 = "tf.Fill"(%cst_6, %cst_5) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %7:8 = tf_device.replicate([%1, %3, %4, %3] as %arg2: tensor<25x5xf32>, [%2, %5, %6, %5] as %arg3: tensor<25x5xf32>) {n = 4 : i32} {
// CHECK   %10 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
// CHECK    %13 = "tf.Identity"(%arg2) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK     tf_device.return %13 : tensor<25x5xf32>
// CHECK   }) : () -> tensor<25x5xf32>
// CHECK   %11 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
// CHECK     %13 = "tf.Identity"(%arg3) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK     tf_device.return %13 : tensor<25x5xf32>
// CHECK   }) : () -> tensor<25x5xf32>
// CHECK   %12:2 = "tf_device.cluster_func"(%10, %11) <{func = @_func}> {_dynamic_arg_index = [], _has_manual_control_dependencies = true, _replication_info = "cluster__train_helper", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], host_compute_core = [], input_sharding_configuration = ["", ""], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["", ""], padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\01\10\02\18\02\22\10\00\00\00\00\00\01\00\00\01\00\00\00\01\01\00\00*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<25x5xf32>, tensor<25x5xf32>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK   tf_device.return %12#0, %12#1 : tensor<128xf32>, tensor<128xf32>
// CHECK }
// CHECK %8 = "tf.Identity"(%7#1) {device = ""} : (tensor<128xf32>) -> tensor<128xf32>
// CHECK %9 = "tf.Identity"(%7#7) {device = ""} : (tensor<128xf32>) -> tensor<128xf32>
// CHECK return %8, %9 : tensor<128xf32>, tensor<128xf32>

// CHECK-LABEL: func.func private @_func(%arg0: tensor<25x5xf32> {mhlo.sharding = ""}, %arg1: tensor<25x5xf32> {mhlo.sharding = ""}) -> (tensor<128xf32> {mhlo.sharding = ""}, tensor<128xf32> {mhlo.sharding = ""}) {
// CHECK: %cst = "tf.Const"()
// CHECK: %0 = "tf.XlaAllReduce"(%arg0, %cst) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<25x5xf32>, tensor<1x4xi32>) -> tensor<25x5xf32>
// CHECK: %cst_0 = "tf.Const"()
// CHECK: %1 = "tf.XlaAllReduce"(%arg1, %cst_0) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<25x5xf32>, tensor<1x4xi32>) -> tensor<25x5xf32>
// CHECK: %cst_1 = "tf.Const"()
// CHECK-NEXT: %cst_2 = "tf.Const"() 
// CHECK: %2 = "tf.Reshape"(%0, %cst_2) : (tensor<25x5xf32>, tensor<1xi32>) -> tensor<125xf32>
// CHECK: %3 = "tf.Pad"(%2, %cst_1) : (tensor<125xf32>, tensor<1x2xi32>) -> tensor<128xf32>
// CHECK: %4 = "tf.Reshape"(%1, %cst_2) : (tensor<25x5xf32>, tensor<1xi32>) -> tensor<125xf32>
// CHECK: %5 = "tf.Pad"(%4, %cst_1) : (tensor<125xf32>, tensor<1x2xi32>) -> tensor<128xf32>
// CHECK: return %3, %5 : tensor<128xf32>, tensor<128xf32>

module attributes {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:CPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:1", "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:1", "/job:tpu_host_worker/replica:0/task:1/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1850 : i32}} {
  func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<25x5xf32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"},
                  %arg1: tensor<*x!tf_type.resource<tensor<25x5xf32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0"})
                  -> (tensor<*xf32>, tensor<*xf32>) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "steps,unknown,unknown_0,unknown_1,unknown_2,unknown_3,unknown_4,unknown_5,unknown_6,unknown_7,unknown_8,unknown_9,unknown_10,unknown_11,unknown_12,unknown_13", outputs = "statefulpartitionedcall_RetVal"}} {
    %0:2 = tf_executor.graph {
      %outputs_0:2, %control_1 = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\00\82\01\05h\01\88\01\01", executor_type = "", f = @_func}> {_collective_manager_ids = [], _read_only_resource_inputs = [8, 9, 10, 11, 12, 13], device = ""} : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>, tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> (tensor<*xf32>,tensor<*xf32>)
      tf_executor.fetch %outputs_0#0, %outputs_0#1 : tensor<*xf32>, tensor<*xf32>
    }
    return %0#0, %0#1: tensor<*xf32>, tensor<*xf32>
  }

  func.func private @_func(%arg0: tensor<!tf_type.resource>, %arg1: tensor<!tf_type.resource>) -> (tensor<*xf32>, tensor<*xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._disable_acd = true, tf.signature.is_stateful} {
    %0:2 = tf_executor.graph {
      %control = tf_executor.island wraps "tf.NoOp"() {_pivot_for_cluster = "cluster__train_helper", device = ""} : () -> ()
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {_has_manual_control_dependencies = true, _tpu_replicate = "cluster__train_helper", device = ""} : () -> ()
      %control_1 = tf_executor.island(%control) wraps "tf.TPUReplicateMetadata"() <{allow_soft_placement = false, computation_shape = [], device_assignment = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 4 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\01\10\02\18\02\22\10\00\00\00\00\00\01\00\00\01\00\00\00\01\01\00\00*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true}> {_has_manual_control_dependencies = true, _tpu_replicate = "cluster__train_helper", device = ""} : () -> ()
      %outputs, %control_2 = tf_executor.island(%control_1) wraps "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster__train_helper", device = ""} : () -> tensor<!tf_type.string>
      %outputs_3, %control_4 = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
      %outputs_5, %control_6 = tf_executor.island wraps "tf.Identity"(%outputs_3) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_7, %control_8 = tf_executor.island wraps "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> {device = ""} : () -> tensor<1xi32>
      %outputs_9, %control_10 = tf_executor.island wraps "tf.Const"() <{value = dense<[[0, 3]]> : tensor<1x2xi32>}> {device = ""} : () -> tensor<1x2xi32>
      %outputs_11, %control_12 = tf_executor.island wraps "tf.Reshape"(%outputs_5, %outputs_7) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1xi32>) -> tensor<*xf32>
      %outputs_13, %control_14 = tf_executor.island wraps "tf.Pad"(%outputs_11, %outputs_9) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1x2xi32>) -> tensor<*xf32>
      %outputs_15, %control_16 = tf_executor.island wraps "tf.Identity"(%outputs_13) {_tpu_output_identity = true, _tpu_replicate = "cluster__train_helper", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_17, %control_18 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
      %outputs_19, %control_20 = tf_executor.island wraps "tf.Identity"(%outputs_17) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_21, %control_22 = tf_executor.island wraps "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> {device = ""} : () -> tensor<1xi32>
      %outputs_23, %control_24 = tf_executor.island wraps "tf.Const"() <{value = dense<[[0, 3]]> : tensor<1x2xi32>}> {device = ""} : () -> tensor<1x2xi32>
      %outputs_25, %control_26 = tf_executor.island wraps "tf.Reshape"(%outputs_19, %outputs_21) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1xi32>) -> tensor<*xf32>
      %outputs_27, %control_28 = tf_executor.island wraps "tf.Pad"(%outputs_25, %outputs_23) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1x2xi32>) -> tensor<*xf32>
      %outputs_29, %control_30 = tf_executor.island wraps "tf.Identity"(%outputs_27) {_tpu_output_identity = true, _tpu_replicate = "cluster__train_helper", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_31:4, %control_32 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_15) {device = ""} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
      %outputs_33:4, %control_34 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_29) {device = ""} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
      %outputs_35, %control_36 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#0) {_has_manual_control_dependencies = true, device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %control_37 = tf_executor.island(%control_36) wraps "tf.NoOp"() {_has_manual_control_dependencies = true, device = ""} : () -> ()
      %outputs_38, %control_39 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_40, %control_41 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_42, %control_43 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#3) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_44, %control_45 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_46, %control_47 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_48, %control_49 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_50, %control_51 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#3) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.fetch %outputs_38, %outputs_50 : tensor<*xf32>, tensor<*xf32>
    }
    return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
  }
}