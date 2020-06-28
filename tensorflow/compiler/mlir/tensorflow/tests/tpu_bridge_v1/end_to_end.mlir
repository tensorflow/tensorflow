// RUN: tf-opt %s -tf-tpu-bridge-v1 | FileCheck %s

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"], tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 296 : i32}} {
  func @main() {
// CHECK: std.constant
// CHECK: TPUCompile
// CHECK: TPUExecute
// CHECK-NOT: func @_func
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "std.constant"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %outputs_0, %control_1 = tf_executor.island wraps "std.constant"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %control_2 = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], name = "TPUReplicateMetadata", num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_tpu = true} : () -> ()
      %outputs_3, %control_4 = tf_executor.island wraps "tf.Placeholder"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "x", shape = "tfshape$dim { }"} : () -> tensor<0xf32>
      %outputs_5, %control_6 = tf_executor.island wraps "tf.TPUReplicatedInput"(%outputs_3) {N = 1 : i64, T = "tfdtype$DT_FLOAT", device = "", name = "input0"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_7, %control_8 = tf_executor.island wraps "tf.Identity"(%outputs_5) {T = "tfdtype$DT_FLOAT", _tpu_input_identity = true, _tpu_replicate = "cluster", device = "", name = "replicated_input_0"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_9, %control_10 = tf_executor.island wraps "tf.Mul"(%outputs_7, %outputs) {T = "tfdtype$DT_FLOAT", _tpu_replicate = "cluster", device = "", name = "mul"} : (tensor<0xf32>, tensor<f32>) -> tensor<0xf32>
      %outputs_11, %control_12 = tf_executor.island wraps "tf.Placeholder"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "y", shape = "tfshape$dim { }"} : () -> tensor<0xf32>
      %outputs_13, %control_14 = tf_executor.island wraps "tf.TPUReplicatedInput"(%outputs_11) {N = 1 : i64, T = "tfdtype$DT_FLOAT", device = "", name = "input1"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_15, %control_16 = tf_executor.island wraps "tf.Identity"(%outputs_13) {T = "tfdtype$DT_FLOAT", _tpu_input_identity = true, _tpu_replicate = "cluster", device = "", name = "replicated_input_1"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_17, %control_18 = tf_executor.island wraps "tf.Mul"(%outputs_15, %outputs_0) {T = "tfdtype$DT_FLOAT", _tpu_replicate = "cluster", device = "", name = "mul_1"} : (tensor<0xf32>, tensor<f32>) -> tensor<0xf32>
      %outputs_19, %control_20 = tf_executor.island wraps "tf.AddV2"(%outputs_9, %outputs_17) {T = "tfdtype$DT_FLOAT", _tpu_replicate = "cluster", device = "", name = "add"} : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
      %outputs_21, %control_22 = tf_executor.island wraps "tf.Identity"(%outputs_19) {T = "tfdtype$DT_FLOAT", _tpu_output_identity = true, _tpu_replicate = "cluster", device = "/device:TPU_REPLICATED_CORE:0", name = "Identity"} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_23, %control_24 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_21) {T = "tfdtype$DT_FLOAT", device = "", name = "output0", num_replicas = 1 : i64} : (tensor<0xf32>) -> tensor<0xf32>
      %outputs_25, %control_26 = tf_executor.island wraps "tf.Identity"(%outputs_23) {T = "tfdtype$DT_FLOAT", device = "", name = "output_0_shard_0"} : (tensor<0xf32>) -> tensor<0xf32>
      %control_27 = tf_executor.island(%control_2, %control_26) wraps "tf.NoOp"() : () -> ()
      tf_executor.fetch %control_27 : !tf_executor.control
    }
    return
  }
}
