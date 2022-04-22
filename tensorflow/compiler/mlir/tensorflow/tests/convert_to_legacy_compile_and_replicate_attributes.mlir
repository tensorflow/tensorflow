// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-convert-to-legacy-compile-and-replicate-attributes | FileCheck %s

// CHECK-LABEL: func @convert_to_legacy_attribute
func.func @convert_to_legacy_attribute(%arg0: tensor<*xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> attributes {tf._disable_call_shape_inference = true, tf.signature.is_stateful} {
  // CHECK-NOT: _replication_info =
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.GuaranteeConst"(%arg1) {T = f32, device = ""} : (tensor<f32>) -> tensor<f32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.GuaranteeConst"(%arg2) {T = f32, device = ""} : (tensor<f32>) -> tensor<f32>
    %control_2 = tf_executor.island wraps "tf.NoOp"() {_pivot_for_cluster = "cluster", device = ""} : () -> ()
    // CHECK: %control_3 = tf_executor.island(%control_2) wraps "tf.NoOp"() {_tpu_replicate = "cluster", device = ""} : () -> ()
    %control_3 = tf_executor.island(%control_2) wraps "tf.NoOp"() {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : () -> ()
    %control_4 = tf_executor.island(%control_2) wraps "tf.TPUReplicateMetadata"() {_replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    %outputs_5, %control_6 = tf_executor.island(%control_4) wraps "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
    %outputs_7, %control_8 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0) {device = "", index = 0 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_9, %control_10 = tf_executor.island(%control_4) wraps "tf.Identity"(%outputs_7) {_replication_info = "cluster", _tpu_input_identity = true, _xla_compile_device_type = "TPU", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.Mul"(%outputs, %outputs_9) {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    %outputs_13, %control_14 = tf_executor.island wraps "tf.AddV2"(%outputs_11, %outputs_0) {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %outputs_15, %control_16 = tf_executor.island wraps "tf.Identity"(%outputs_13) {_replication_info = "cluster", _tpu_output_identity = true, _xla_compile_device_type = "TPU", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_17, %control_18 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_15) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_19, %control_20 = tf_executor.island(%control_3) wraps "tf.Identity"(%outputs_17) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs_19 : tensor<*xf32>
  }
  func.return %0 : tensor<*xf32>
}

// -----

func.func @convert_to_legacy_attributes_failure(%arg0: tensor<*xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> attributes {tf._disable_call_shape_inference = true, tf.signature.is_stateful} {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.GuaranteeConst"(%arg1) {T = f32, device = ""} : (tensor<f32>) -> tensor<f32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.GuaranteeConst"(%arg2) {T = f32, device = ""} : (tensor<f32>) -> tensor<f32>
    %control_2 = tf_executor.island wraps "tf.NoOp"() {_pivot_for_cluster = "cluster", device = ""} : () -> ()
    %control_3 = tf_executor.island(%control_2) wraps "tf.NoOp"() {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : () -> ()
    %control_4 = tf_executor.island(%control_2) wraps "tf.TPUReplicateMetadata"() {_replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    %outputs_5, %control_6 = tf_executor.island(%control_4) wraps "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
    %outputs_7, %control_8 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0) {device = "", index = 0 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_9, %control_10 = tf_executor.island(%control_4) wraps "tf.Identity"(%outputs_7) {_replication_info = "cluster", _tpu_input_identity = true, _xla_compile_device_type = "TPU", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.Mul"(%outputs, %outputs_9) {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    %outputs_13, %control_14 = tf_executor.island wraps "tf.AddV2"(%outputs_11, %outputs_0) {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // expected-error @+1 {{'tf.Identity' op has '_replication_info' attribute but not '_xla_compile_device_type' attribute which is unsupported}}
    %outputs_15, %control_16 = tf_executor.island wraps "tf.Identity"(%outputs_13) {_replication_info = "cluster", _tpu_output_identity = true, device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_17, %control_18 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_15) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %outputs_19, %control_20 = tf_executor.island(%control_3) wraps "tf.Identity"(%outputs_17) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs_19 : tensor<*xf32>
  }
  func.return %0 : tensor<*xf32>
}
