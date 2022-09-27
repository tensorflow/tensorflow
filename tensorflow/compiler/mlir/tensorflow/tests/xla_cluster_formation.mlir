// RUN: tf-opt %s -tf-xla-cluster-formation | FileCheck %s

// Check that we outline the partitioned call to a device cluster (since it has
// `_xla_compile_device_type`).
// CHECK: tf_device.cluster
// CHECK-NEXT: tf.StatefulPartitionedCall
// CHECK-NEXT: tf_device.return
// CHECK: tf.Const
// CHECK: tf.Add
func.func @xla_must_compile_true(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", _xla_compile_device_type = "CPU", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

// Check that we don't outline the partitioned call to a device cluster (since
// it does not has `_xla_compile_device_type`).
// CHECK-NOT: tf_device.cluster
func.func @xla_must_compile_false(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
