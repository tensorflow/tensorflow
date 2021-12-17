// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-clustering="oplist=all min-cluster-size=2"                 \
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

// CHECK-LABEL: func @do_not_cluster_hoistable_ops
func @do_not_cluster_hoistable_ops(
    %arg0 : tensor<i32>,
    %arg1 : tensor<*x!tf_type.resource>,
    %arg2 : tensor<*x!tf_type.resource>
  ) -> tensor<i32> {
  // CHECK: "tf.Const"
  // CHECK: "tf.ReadVariableOp"
  // CHECK: "tf.ReadVariableOp"
  // CHECK: "tf.Add"
  // CHECK: "tf.Neg"
  // CHECK: "tf.Sub"
  %c = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %x = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> tensor<i32>
  %y = "tf.ReadVariableOp"(%arg2) : (tensor<*x!tf_type.resource>) -> tensor<i32>
  %0 = "tf.Add"(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%0, %c) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[CLUSTER:.*]] = "tf_device.cluster"()
  // CHECK:                 "tf.Sub"
  // CHECK:                 "tf.Neg"
  // CHECK:   %[[RET:.*]] = "tf.Add"
  // CHECK:   tf_device.return %[[RET]]
  %3 = "tf.Sub"(%arg0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
  %5 = "tf.Add"(%2, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: }) {policy = "tfrt.auto-fusion"}
  // CHECK: return %[[CLUSTER]]
  return %5 : tensor<i32>
}
