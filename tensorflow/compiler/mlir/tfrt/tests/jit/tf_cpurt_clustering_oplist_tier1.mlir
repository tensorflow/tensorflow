// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=tier1 min-cluster-size=2"               \
// RUN: | FileCheck %s --check-prefix CHECK --check-prefix=TIER1
// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=tier1metadata min-cluster-size=2"       \
// RUN: | FileCheck %s --check-prefix CHECK --check-prefix=METADATA
// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=tier1reductions min-cluster-size=2"     \
// RUN: | FileCheck %s --check-prefix CHECK --check-prefix=REDUCTIONS

// CHECK-LABEL: func @single_cluster_one_result
func @single_cluster_one_result(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>)
    -> tensor<?xi32> {
  // CHECK: %[[CLUSTER:.*]] = "tf_device.cluster"()
  // TIER1-NOT:             "tf.Sum"
  // TIER1:                 "tf.Add"
  // TIER1:                 "tf.Neg"
  // TIER1:                 "tf.Sub"
  // TIER1:                 "tf.Neg"
  // TIER1:   %[[RET:.*]] = "tf.Add"
  // TIER1:   tf_device.return %[[RET]]

  // METADATA-NOT:             "tf.Sum"
  // METADATA:                 "tf.Add"
  // METADATA:                 "tf.Neg"
  // METADATA:                 "tf.Sub"
  // METADATA:                 "tf.Neg"
  // METADATA:                 "tf.Add"
  // METADATA:   %[[RET:.*]] = "tf.Shape"
  // METADATA:   tf_device.return %[[RET]]

  // REDUCTIONS:                 "tf.Sum"
  // REDUCTIONS:                 "tf.Add"
  // REDUCTIONS:                 "tf.Neg"
  // REDUCTIONS:                 "tf.Sub"
  // REDUCTIONS:                 "tf.Neg"
  // REDUCTIONS:   %[[RET:.*]] = "tf.Add"
  // REDUCTIONS:   tf_device.return %[[RET]]
  %dimension = "tf.Const"() { value = dense<0> : tensor<1xi64> } : () -> tensor<1xi64>
  %s = "tf.Sum"(%arg0, %dimension) { keep_dims = false }: (tensor<?xi32>, tensor<1xi64>) -> tensor<i32>
  %0 = "tf.Add"(%s, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%s, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %5 = "tf.Shape"(%4) : (tensor<i32>) -> tensor<?xi32>
  // CHECK: }) {policy = "tfrt.auto-fusion"}
  // TIER1: %[[SHAPE:.*]] = "tf.Shape"(%[[CLUSTER]])
  // TIER1: return %[[SHAPE]]

  // REDUCTIONS: %[[SHAPE:.*]] = "tf.Shape"(%[[CLUSTER]])
  // REDUCTIONS: return %[[SHAPE]]
  return %5 : tensor<?xi32>
}
