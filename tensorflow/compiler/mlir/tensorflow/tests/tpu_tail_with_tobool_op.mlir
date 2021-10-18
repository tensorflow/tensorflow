// RUN: tf-opt %s -tf-tpu-bridge 2>&1 | FileCheck %s

// This test verifies that the tail extraction is not terminated prematurely
// in handling tf.If op which would end up with excessive host-device
// communication.

// In this test, all ops other than tf.Rank are marked with outside_compilation
// . So the TPU program should contain tf.Rank op and there should be no
// host-device communication.
// CHECK: tf._TPUCompileMlir
// CHECK-SAME: tf.Rank
// CHECK-NOT: tf._XlaHostComputeMlir
// CHECK-NOT: tf._XlaRecvAtHost
// CHECK-NOT: tf._XlaSendFromHost

module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 741 : i32}}  {
  func private @if_then_branch(%arg0: tensor<*xi1>, %arg1: tensor<*xi64>, %arg2: tensor<*xi64>) -> tensor<*xi1> {
    return %arg0 : tensor<*xi1>
  }
  func private @if_else_branch(%arg0: tensor<*xi1>, %arg1: tensor<*xi64>, %arg2: tensor<*xi64>) -> tensor<*xi1> {
    return %arg0 : tensor<*xi1>
  }
  func @"tpu_subgraph"(%arg0: tensor<*xi64>) -> tensor<*xi1> {
    %cst = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %cst_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    %0 = "tf.Equal"(%arg0, %cst) {_tpu_replicate = "cluster", device = "", incompatible_shape_error = true} : (tensor<*xi64>, tensor<i64>) -> tensor<*xi1>
    %1 = "tf.Rank"(%0) {_tpu_replicate = "cluster", device = ""} : (tensor<*xi1>) -> tensor<*xi32>
    %2 = "tf.Range"(%cst_0, %1, %cst_1) {_tpu_replicate = "cluster", _xla_outside_compilation = "0", device = ""} : (tensor<i32>, tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %3 = "tf.All"(%0, %2) {_tpu_replicate = "cluster", _xla_outside_compilation = "0", device = "", keep_dims = false} : (tensor<*xi1>, tensor<*xi32>) -> tensor<*xi1>
    %4 = "tf.If"(%3, %3, %arg0, %cst) {_tpu_replicate = "cluster", _xla_outside_compilation = "0", device = "", else_branch = @if_else_branch, is_stateless = false, then_branch = @if_then_branch} : (tensor<*xi1>, tensor<*xi1>, tensor<*xi64>, tensor<i64>) -> tensor<*xi1>
    return %4 : tensor<*xi1>
  }
}
