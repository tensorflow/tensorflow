// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=tf.Add,tf.Sub,tf.Neg min-cluster-size=2"\
// RUN: | FileCheck %s

// CHECK-LABEL: func @single_cluster_one_result
func @single_cluster_one_result(%arg0 : tensor<i32>, %arg1 : tensor<i32>)
    -> tensor<i32> {
  // CHECK: %[[CLUSTER:.*]] = "tf_device.cluster"()
  // CHECK:                 "tf.Add"
  // CHECK:                 "tf.Neg"
  // CHECK:                 "tf.Sub"
  // CHECK:                 "tf.Neg"
  // CHECK:   %[[RET:.*]] = "tf.Add"
  // CHECK:   tf_device.return %[[RET]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: }) {policy = "tfrt.auto-fusion"}
  // CHECK: return %[[CLUSTER]]
  return %4 : tensor<i32>
}
