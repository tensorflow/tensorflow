// RUN: tf-opt %s -tf-canonicalize-compile-and-replicate-attributes | FileCheck %s


module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"], tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 296 : i32}} {
  // CHECK-LABEL: func @main
  func.func @main() {
    tf_executor.graph {
      // CHECK: tf_executor.island wraps "tf.TPUReplicateMetadata"() {_replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], name = "TPUReplicateMetadata", num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
      %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], name = "TPUReplicateMetadata", num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_tpu = true, use_spmd_for_xla_partitioning = false} : () -> ()
      %outputs_0, %control_0 = tf_executor.island wraps "tf.Placeholder"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "y", shape = "tfshape$dim { }"} : () -> tensor<0xf32>
      %outputs_1, %control_1 = tf_executor.island wraps "tf.TPUReplicatedInput"(%outputs_0) {N = 1 : i64, T = "tfdtype$DT_FLOAT", device = "", name = "input1"} : (tensor<0xf32>) -> tensor<0xf32>
      // CHECK: tf_executor.island wraps "tf.Identity"(%outputs_1) {T = "tfdtype$DT_FLOAT", _replication_info = "cluster", _tpu_input_identity = true, _xla_compile_device_type = "TPU", device = "", name = "replicated_input_1"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_2, %control_2 = tf_executor.island wraps "tf.Identity"(%outputs_1) {T = "tfdtype$DT_FLOAT", _tpu_input_identity = true, _tpu_replicate = "cluster", device = "", name = "replicated_input_1"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_3, %control_3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_2) {T = "tfdtype$DT_FLOAT", device = "", name = "output0", num_replicas = 1 : i64} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_4, %control_4 = tf_executor.island wraps "tf.Identity"(%outputs_3) {T = "tfdtype$DT_FLOAT", device = "", name = "output_0_shard_0"} : (tensor<0xf32>) -> tensor<0xf32>
      %control_5 = tf_executor.island(%control, %control_4) wraps "tf.NoOp"() : () -> ()
      tf_executor.fetch %control_5 : !tf_executor.control
    }
    func.return
  }
}
