// Note: 'oplist=all', and not 'oplist=all(,all)+', is special-cased.
// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=all,all min-cluster-size=2"             \
// RUN: | FileCheck %s

// CHECK-LABEL: func @no_cluster_because_of_all_all
func @no_cluster_because_of_all_all(%arg0 : tensor<i32>, %arg1 : tensor<i32>)
    -> tensor<i32> {
  // CHECK-NOT:             "tf_device.cluster"()
  // CHECK:                 "tf.Add"
  // CHECK:                 "tf.Neg"
  // CHECK:                 "tf.Sub"
  // CHECK:                 "tf.Neg"
  // CHECK:   %[[RET:.*]] = "tf.Add"
  // CHECK-NOT:   tf_device.return %[[RET]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NOT: }) {policy = "tfrt.auto-fusion"}
  // CHECK: return %[[RET]]
  return %4 : tensor<i32>
}
