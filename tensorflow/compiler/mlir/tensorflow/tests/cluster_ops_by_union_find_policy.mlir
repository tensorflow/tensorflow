// RUN: tf-opt -cluster-ops-by-policy="oplist=tf.Add,tf.Sub,tf.Neg algorithm=union-find min-cluster-size=2" %s | FileCheck %s

// CHECK-LABEL: func @single_cluster
func @single_cluster(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
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
  // CHECK: })
  // CHECK: return %[[CLUSTER]]
  return %4 : tensor<i32>
}

// CHECK-LABEL: func @single_cluster_with_return
func @single_cluster_with_return(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: %[[CLUSTER:.*]]:2 = "tf_device.cluster"()
  // CHECK:                  "tf.Add"
  // CHECK:   %[[RET0:.*]] = "tf.Neg"
  // CHECK:                  "tf.Sub"
  // CHECK:                  "tf.Neg"
  // CHECK:   %[[RET1:.*]] = "tf.Add"
  // CHECK:   tf_device.return %[[RET0]], %[[RET1]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[CLUSTER]]#0, %[[CLUSTER]]#1
  return %1, %4 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @unsupported_op_breaks_cluster_0
func @unsupported_op_breaks_cluster_0(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: %[[CLUSTER_0:.*]] = "tf_device.cluster"()
  // CHECK:                 "tf.Add"
  // CHECK:   %[[RET:.*]] = "tf.Neg"
  // CHECK:   tf_device.return %[[RET]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[CLUSTER_1:.*]] = "tf_device.cluster"()
  // CHECK:                 "tf.Sub"
  // CHECK:   %[[RET:.*]] = "tf.Neg"
  // CHECK:   tf_device.return %[[RET]]
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[CLUSTER_0]], %[[CLUSTER_1]])
  %4 = "tf.Mul"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[NEG:.*]] = "tf.Neg"(%[[MUL]])
  %5 = "tf.Neg"(%4) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[CLUSTER_0]], %[[NEG]]
  return %1, %5 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @unsupported_op_breaks_cluster_1
func @unsupported_op_breaks_cluster_1(%arg0 : tensor<i32>) -> tensor<i32> {
  // CHECK: %[[CLUSTER:.*]]:2 = "tf_device.cluster"()
  // CHECK:   %[[RET0:.*]] = "tf.Neg"
  // CHECK:   %[[RET1:.*]] = "tf.Neg"
  // CHECK:   tf_device.return %[[RET0]], %[[RET1]]
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[CLUSTER]]#1, %[[CLUSTER]]#1)
  %2 = "tf.Mul"(%1, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[ADD:.*]] = "tf.Add"(%[[CLUSTER]]#0, %[[MUL]])
  %3 = "tf.Add"(%0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %3 : tensor<i32>
}
