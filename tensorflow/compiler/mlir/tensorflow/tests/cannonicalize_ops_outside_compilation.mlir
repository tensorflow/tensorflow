// RUN: tf-opt %s -tf-tpu-bridge 2>&1 | FileCheck %s

// This test verifies that the tail extraction is not terminated prematurely
// due to the outside compilation attribute could be removed in
// canonicalization of Reshape ops.

// Reshape should not be executed on TPU as all are marked by outside
// compilation. And there should be no host-device communication.
// CHECK: tf._TPUCompileMlir
// CHECK-NOT: tf.Reshape
// CHECK-NOT: tf._XlaHostComputeMlir

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 772 : i32}}  {
  func.func @inference_Fn_101680(%arg0: tensor<*x!tf_type.string>, %arg1: tensor<*x!tf_type.string>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> tensor<*xi32> {
    "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster_Fn", allow_soft_placement = true, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    %0 = "tf.Shape"(%arg0) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "1", device = ""} : (tensor<*x!tf_type.string>) -> tensor<*xi32>
    %1 = "tf.Shape"(%arg1) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "0", device = ""} : (tensor<*x!tf_type.string>) -> tensor<*xi32>
    %2:3 = "tf.UnpackHyp"(%arg1) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "0", device = "", max_seq_length = 16 : i64} : (tensor<*x!tf_type.string>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xf32>)
    %3 = "tf.Reshape"(%2#2, %1) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "0", device = ""} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %4 = "tf.Reshape"(%3, %0) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "1", device = ""} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %5 = "tf.TPUReplicatedOutput"(%4) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %6 = "tf.Reshape"(%2#0, %arg2) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "1", device = ""} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %7 = "tf.Reshape"(%2#1, %arg3) {_tpu_replicate = "cluster_Fn", _xla_outside_compilation = "1", device = ""} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %7 : tensor<*xi32>
  }
}
