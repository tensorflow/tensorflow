// RUN: dtensor-opt %s -dtensor-tpu-integration | FileCheck %s

// Test that tf_device.Cluster op is created for tf.StatefulPartitionedCall that
// runs on TPU's.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0, %1 = "tf.StatefulPartitionedCall"(%arg0) {config = "|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @tpu_func} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)

  %2, %3 = "tf.StatefulPartitionedCall"(%arg0) {config = "|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @cpu_func} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @tpu_func
func.func @tpu_func(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   tf.Const
  // CHECK-NEXT:   tf.Add
  // CHECK-NEXT:   tf_device.return
  // CHECK:      _tpu_replicate
  // CHECK-SAME: device_assignment = []
  // CHECK-SAME: num_cores_per_replica = 1
  // CHECK-SAME: padding_map = []
  // CHECK-SAME: step_marker_location = ""
  // CHECK-SAME: topology = ""
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%1, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2, %arg0: tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @cpu_func
func.func @cpu_func(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK-NOT:      tf_device.Cluster
  %0, %1 = "tf.A"(%arg0) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1: tensor<i32>, tensor<i32>
}
