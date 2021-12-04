// RUN: xla-opt -split-input-file -verify-diagnostics -xla-legalize-tf-collective %s | FileCheck %s

// CHECK-LABEL: func @all_reduce_cross_replica
func @all_reduce_cross_replica(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[1],[2]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK{LITERAL}: replica_groups = dense<[[1], [2]]> : tensor<2x1xi64>
  // CHECK-NOT: channel_handle
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @all_reduce_cross_replica_and_partition
func @all_reduce_cross_replica_and_partition(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[1],[2]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 1 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[1], [2]]> : tensor<2x1xi64>
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 2 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[1], [2]]> : tensor<2x1xi64>
  %1 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
