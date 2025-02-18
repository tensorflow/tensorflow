// RUN: tf-opt %s -split-input-file -tf-xla-broadcast | FileCheck %s
module attributes {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:CPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:1", "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:1", "/job:tpu_host_worker/replica:0/task:1/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1850 : i32}} {
// CHECK-LABEL: func @move_broadcast_non_spmd
func.func @move_broadcast_non_spmd(%arg0: tensor<f32>) -> () {
  // CHECK:      %[[ELEM_0:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[SHAPE_0:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[FULL_0:.*]] = "tf.Fill"(%[[SHAPE_0]], %[[ELEM_0]]) {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK:      %[[ELEM_1:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[SHAPE_1:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[FULL_1:.*]] = "tf.Fill"(%[[SHAPE_1]], %[[ELEM_1]]) {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: tf_device.replicate([%arg0, %[[FULL_0]], %[[FULL_1]], %[[FULL_0]]] as %[[REPVAR:.*]]: tensor<f32>) {n = 4 : i32} {
  // CHECK-NEXT:   %[[ID:.*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
  // CHECK-NEXT:     %[[IDINSIDE:.*]] = "tf.Identity"(%[[REPVAR]]) {_ici_weight_distribution_mlir_bridge_marker = true} : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     tf_device.return %[[IDINSIDE]] : tensor<f32>
  // CHECK-NEXT:   }) {_ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
  // CHECK-NEXT:   "tf_device.cluster"() ({
  // CHECK-NEXT:     %[[GROUP:.*]] = "tf.Const"()
  // CHECK-SAME:       [0, 1, 2, 3]
  // CHECK-NEXT:     %[[REDUCED:.*]] = "tf.XlaAllReduce"(%[[ID]], %[[GROUP]]) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<f32>, tensor<1x4xi32>) -> tensor<f32>
  // CHECK-NEXT:     "tf.OpA"(%[[REDUCED]]) : (tensor<f32>) -> ()
  tf_device.replicate {n = 4 : i32} {
    "tf_device.cluster"() ({
      "tf.OpA"(%arg0) : (tensor<f32>) -> ()
      tf_device.return
    }) {allow_soft_placement = false, computation_shape = [], device_assignment = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 4 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\01\10\02\18\02\22\10\00\00\00\00\00\01\00\00\01\00\00\00\01\01\00\00*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
  }
  func.return
}
}

// -----
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2", "/job:worker/replica:0/task:0/device:TPU:3", "/job:worker/replica:0/task:0/device:TPU:4", "/job:worker/replica:0/task:0/device:TPU:5", "/job:worker/replica:0/task:0/device:TPU:6", "/job:worker/replica:0/task:0/device:TPU:7"]} {
// CHECK-LABEL: func @move_broadcast_spmd
func.func @move_broadcast_spmd(%arg0: tensor<f32>) -> () {
  // CHECK:      %[[ELEM_0:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[SHAPE_0:.*]] = "tf.Const"()
  // CHECK:      {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: %[[FULL_0:.*]] = "tf.Fill"(%[[SHAPE_0]], %[[ELEM_0]]) {_ici_weight_distribution_mlir_bridge_marker = true}
  // CHECK-NEXT: tf_device.replicate([%arg0, %[[FULL_0]], %[[FULL_0]], %[[FULL_0]]] as %[[REPVAR:.*]]: tensor<f32>) {n = 4 : i32} {
  // CHECK-NEXT:   %[[ID:.*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
  // CHECK-NEXT:     %[[IDINSIDE:.*]] = "tf.Identity"(%[[REPVAR]]) {_ici_weight_distribution_mlir_bridge_marker = true} : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     tf_device.return %[[IDINSIDE]] : tensor<f32>
  // CHECK-NEXT:   }) {_ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<f32>
  // CHECK-NEXT:   "tf_device.cluster"() ({
  // CHECK-NEXT:     %[[GROUP:.*]] = "tf.Const"()
  // CHECK-SAME:       [0, 1, 2, 3]
  // CHECK-NEXT:     %[[REDUCED:.*]] = "tf.XlaAllReduce"(%[[ID]], %[[GROUP]]) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<f32>, tensor<1x4xi32>) -> tensor<f32>
  // CHECK-NEXT:     "tf.OpA"(%[[REDUCED]]) : (tensor<f32>) -> ()
  tf_device.replicate {n = 4 : i32} {
    "tf_device.cluster"() ({
      "tf.OpA"(%arg0) : (tensor<f32>) -> ()
      tf_device.return
    }) {allow_soft_placement = false, computation_shape = [], device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], host_compute_core = [], num_cores_per_replica = 2 : i64, num_replicas = 4 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
  }
  func.return
}
}
