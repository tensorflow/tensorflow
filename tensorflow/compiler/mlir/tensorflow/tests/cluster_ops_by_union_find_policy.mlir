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

// CHECK-LABEL: func @unsupported_op_breaks_cluster
func @unsupported_op_breaks_cluster(%arg0 : tensor<i32>) -> tensor<i32> {
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

// CHECK-LABEL: func @single_cluster_from_independent_ops
func @single_cluster_from_independent_ops(%arg0 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: %[[CLUSTER:.*]]:2 = "tf_device.cluster"()
  // CHECK: %[[NEG0:.*]] = "tf.Neg"(%arg0)
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[NEG1:.*]] = "tf.Neg"(%arg0)
  %1 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[ADD0:.*]] = "tf.Add"(%[[NEG0]], %arg0)
  %2 = "tf.Add"(%0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[ADD1:.*]] = "tf.Add"(%[[NEG1]], %arg0)
  %3 = "tf.Add"(%1, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: tf_device.return %[[ADD0]], %[[ADD1]]
  // CHECK: return %[[CLUSTER]]#0, %[[CLUSTER]]#1
  return %2, %3 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @single_cluster_from_independent_ops_and_unsupported_op_breaks_cluster
func @single_cluster_from_independent_ops_and_unsupported_op_breaks_cluster(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: %[[CLUSTER:.*]]:2 = "tf_device.cluster"()
  // CHECK: %[[ADD:.*]] = "tf.Add"(%arg0, %arg1)
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[NEG0:.*]] = "tf.Neg"(%[[ADD]])
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %arg1)
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[NEG1:.*]] = "tf.Neg"(%[[SUB]])
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK:   tf_device.return %[[NEG0]], %[[NEG1]]
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[CLUSTER]]#0, %[[CLUSTER]]#1)
  %4 = "tf.Mul"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[NEG3:.*]] = "tf.Neg"(%[[MUL]])
  %5 = "tf.Neg"(%4) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[CLUSTER]]#0, %[[NEG3]]
  return %1, %5 : tensor<i32>, tensor<i32>
}


// CHECK-LABEL: func @cluster_test
func @cluster_test(%arg0 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Not"(%0) : (tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%1, %2) :  (tensor<i32>, tensor<i32>) -> tensor<i32>
  %5 = "tf.Add"(%4, %3) :  (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %5 : tensor<i32>
}
