// RUN: tf-opt %s -tf-xla-cluster-formation | FileCheck %s

// CHECK-LABEL: func @simple_stateful_partitioned_call
// CHECK: tf_device.cluster
// CHECK-NEXT: tf.StatefulPartitionedCall
// CHECK-NEXT: tf_device.return
// CHECK: tf.Const
// CHECK: tf.Add
func.func @simple_stateful_partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
