// RUN: tf-opt %s -tf-replicated-clustering-bridge-v2 -tfrt-lower-cluster-to-runtime-ops-tpu -tf-dialect-to-executor-v2 --mlir-print-ir-before-all --mlir-print-ir-after-all | FileCheck %s


// CHECK-LABEL: func.func @main
// CHECK %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> tensor<25x5xf32>
// CHECK %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> tensor<25x5xf32>
// CHECK %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %outputs_6, %control_7 = tf_executor.island wraps "tf.Fill"(%outputs_4, %outputs_2) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %outputs_8, %control_9 = tf_executor.island wraps "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %outputs_10, %control_11 = tf_executor.island wraps "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %outputs_12, %control_13 = tf_executor.island wraps "tf.Fill"(%outputs_10, %outputs_8) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %outputs_14, %control_15 = tf_executor.island wraps "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %outputs_16, %control_17 = tf_executor.island wraps "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %outputs_18, %control_19 = tf_executor.island wraps "tf.Fill"(%outputs_16, %outputs_14) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %outputs_20, %control_21 = tf_executor.island wraps "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
// CHECK %outputs_22, %control_23 = tf_executor.island wraps "tf.Const"() <{value = dense<[25, 5]> : tensor<2xi64>}> {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<2xi64>
// CHECK %outputs_24, %control_25 = tf_executor.island wraps "tf.Fill"(%outputs_22, %outputs_20) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<2xi64>, tensor<f32>) -> tensor<25x5xf32>
// CHECK %outputs_26:2, %control_27 = tf_executor.island wraps "tf._TPUCompileMlir"()
// CHECK %outputs_28, %control_29 = tf_executor.island wraps "tf.Identity"(%outputs_26#0) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
// CHECK %control_30 = tf_executor.island wraps "tf.TPUCompileSucceededAssert"(%outputs_28) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.string>) -> ()
// CHECK %outputs_31, %control_32 = tf_executor.island wraps "tf.Identity"(%outputs) {_parallel_execution_ids = "r0:0", device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_33, %control_34 = tf_executor.island wraps "tf.Identity"(%outputs_0) {_parallel_execution_ids = "r0:0", device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_35:2, %control_36 = tf_executor.island wraps "tf.TPUExecute"(%outputs_31, %outputs_33, %outputs_26#1) {_parallel_execution_ids = "r0:0", device = "/job:tpu_host_worker/replica:0/task:0/device:TPU:0"} : (tensor<25x5xf32>, tensor<25x5xf32>, tensor<3x!tf_type.string>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK %outputs_37, %control_38 = tf_executor.island wraps "tf.Identity"(%outputs_6) {_parallel_execution_ids = "r0:1", device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_39, %control_40 = tf_executor.island wraps "tf.Identity"(%outputs_18) {_parallel_execution_ids = "r0:1", device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_41:2, %control_42 = tf_executor.island wraps "tf.TPUExecute"(%outputs_37, %outputs_39, %outputs_26#1) {_parallel_execution_ids = "r0:1", device = "/job:tpu_host_worker/replica:0/task:1/device:TPU:0"} : (tensor<25x5xf32>, tensor<25x5xf32>, tensor<3x!tf_type.string>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK %outputs_43, %control_44 = tf_executor.island wraps "tf.Identity"(%outputs_12) {_parallel_execution_ids = "r0:2", device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_45, %control_46 = tf_executor.island wraps "tf.Identity"(%outputs_24) {_parallel_execution_ids = "r0:2", device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_47:2, %control_48 = tf_executor.island wraps "tf.TPUExecute"(%outputs_43, %outputs_45, %outputs_26#1) {_parallel_execution_ids = "r0:2", device = "/job:tpu_host_worker/replica:0/task:0/device:TPU:1"} : (tensor<25x5xf32>, tensor<25x5xf32>, tensor<3x!tf_type.string>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK %outputs_49, %control_50 = tf_executor.island wraps "tf.Identity"(%outputs_6) {_parallel_execution_ids = "r0:3", device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_51, %control_52 = tf_executor.island wraps "tf.Identity"(%outputs_18) {_parallel_execution_ids = "r0:3", device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", ici_weight_distribution_mlir_bridge_marker = true} : (tensor<25x5xf32>) -> tensor<25x5xf32>
// CHECK %outputs_53:2, %control_54 = tf_executor.island wraps "tf.TPUExecute"(%outputs_49, %outputs_51, %outputs_26#1) {_parallel_execution_ids = "r0:3", device = "/job:tpu_host_worker/replica:0/task:1/device:TPU:1"} : (tensor<25x5xf32>, tensor<25x5xf32>, tensor<3x!tf_type.string>) -> (tensor<128xf32>, tensor<128xf32>)
// CHECK %outputs_55, %control_56 = tf_executor.island wraps "tf.Identity"(%outputs_41#0) {device = "/job:tpu_host_worker/replica:0/task:1/device:TPU:0"} : (tensor<128xf32>) -> tensor<128xf32>
// CHECK %outputs_57, %control_58 = tf_executor.island wraps "tf.Identity"(%outputs_53#1) {device = "/job:tpu_host_worker/replica:0/task:1/device:TPU:1"} : (tensor<128xf32>) -> tensor<128xf32>
// CHECK tf_executor.fetch %outputs_55, %outputs_57, %control, %control_1, %control_36, %control_42, %control_48, %control_54 : tensor<128xf32>, tensor<128xf32>, !tf_executor.control, !tf_executor.control, !tf_executor.control, !tf_executor.control, !tf_executor.control, !tf_executor.control

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