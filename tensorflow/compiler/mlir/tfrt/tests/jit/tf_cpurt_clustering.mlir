// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -tf-cpurt-test-clustering="min-cluster-size=1" -verify-diagnostics   \
// RUN: | FileCheck %s

// CHECK-LABEL: func @no_clusters
func @no_clusters(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.UnknownOp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

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
  // CHECK: })
  // CHECK: return %[[CLUSTER]]
  return %4 : tensor<i32>
}

// CHECK-LABEL: func @single_cluster_two_results
func @single_cluster_two_results(%arg0 : tensor<i32>, %arg1 : tensor<i32>)
    -> (tensor<i32>, tensor<i32>) {
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
  // CHECK: %[[UNS:.*]] = "tf.Unsupported"(%[[CLUSTER]]#1, %[[CLUSTER]]#1)
  %2 = "tf.Unsupported"(%1, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[ADD:.*]] = "tf.Add"(%[[CLUSTER]]#0, %[[UNS]])
  %3 = "tf.Add"(%0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %3 : tensor<i32>
}

// CHECK-LABEL: func @single_cluster_from_independent_ops
func @single_cluster_from_independent_ops(%arg0 : tensor<i32>)
    -> (tensor<i32>, tensor<i32>) {
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
func @single_cluster_from_independent_ops_and_unsupported_op_breaks_cluster(
    %arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: %[[CLUSTER0:.*]]:2 = "tf_device.cluster"()
  // CHECK:   %[[ADD:.*]] = "tf.Add"(%arg0, %arg1)
  // CHECK:   %[[NEG0:.*]] = "tf.Neg"(%[[ADD]])
  // CHECK:   %[[SUB:.*]] = "tf.Sub"(%arg0, %arg1)
  // CHECK:   %[[NEG1:.*]] = "tf.Neg"(%[[SUB]])
  // CHECK:   tf_device.return %[[NEG0]], %[[NEG1]]
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK: %[[UNS:.*]] = "tf.Unsupported"(%[[CLUSTER0]]#0, %[[CLUSTER0]]#1)
  %4 = "tf.Unsupported"(%1, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[CLUSTER1:.*]] = "tf_device.cluster"()
  // CHECK:   %[[NEG3:.*]] = "tf.Neg"(%[[UNS]])
  // CHECK:   tf_device.return %[[NEG3]]
  %5 = "tf.Neg"(%4) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[CLUSTER0]]#0, %[[CLUSTER1]]
  return %1, %5 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @transpose_constraint_propagation
// expected-remark@below {{input #0 constrained to: shape}}
func @transpose_constraint_propagation(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // CHECK: %[[CLUSTER:.*]] = "tf_device.cluster"()
  // CHECK:                 "tf.Shape"
  // CHECK:   %[[RET:.*]] = "tf.Transpose"
  // CHECK:   tf_device.return %[[RET]]
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>
  %1 = "tf.Transpose"(%arg0, %0)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // CHECK: return %[[CLUSTER]]
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @transpose_in_two_clusters
func @transpose_in_two_clusters(%arg0 : tensor<?x?xf32>,
                                %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK: tf.Const
  %0 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32> }
       : () -> tensor<2xi32>

  // CHECK:     %[[CLUSTER_0:.*]] = "tf_device.cluster"()
  // CHECK-NOT:   tf.Const
  // CHECK:       tf.Transpose
  // CHECK:       tf.Rsqrt
  %1 = "tf.Transpose"(%arg0, %0)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2 = "tf.Rsqrt"(%1): (tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK:     %[[CLUSTER_1:.*]] = "tf_device.cluster"()
  // CHECK-NOT:   tf.Const
  // CHECK:       tf.Transpose
  // CHECK:       tf.Rsqrt
  %4 = "tf.Transpose"(%arg1, %0)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %5 = "tf.Rsqrt"(%4): (tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: return %[[CLUSTER_0]], %[[CLUSTER_1]]
  return %2, %5 : tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: func @do_not_cluster_i1_arguments
func @do_not_cluster_i1_arguments(%arg0 : tensor<?xi1>, %arg1 : tensor<?xi1>,
                                  %arg2 : tensor<?xf32>, %arg3 : tensor<?xf32>)
    -> tensor<?xf32> {
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.LogicalOr"(%arg0, %arg1)
        : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
  %1 = "tf.Select"(%0, %arg2, %arg3)
        : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func @do_not_cluster_i1_in_the_body
func @do_not_cluster_i1_in_the_body(%arg0 : tensor<?xf32>,
                                    %arg1 : tensor<?xf32>)
    -> tensor<?xi1> {
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.Less"(%arg0, %arg1): (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

// CHECK-LABEL: func @do_not_cluster_ui64_arguments
func @do_not_cluster_ui64_arguments(%arg0: tensor<?xui64>) -> tensor<?xi64> {
  // CHECK-NOT: tf_device.cluster
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<?xui64>) -> tensor<?xi64>
  tf_device.return %0 : tensor<?xi64>
}
