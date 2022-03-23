// RUN: xla-opt -split-input-file -verify-diagnostics -xla-legalize-tf-collective %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @all_reduce_cross_replica
func @all_reduce_cross_replica(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  // CHECK-NOT: channel_handle
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @all_reduce_cross_replica_and_partition
func @all_reduce_cross_replica_and_partition(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 1 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 2 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  %1 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 1
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @collective_reduce_v2
func @collective_reduce_v2(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 1 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<> : tensor<0x0xi64>
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 2 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<> : tensor<0x0xi64>
  %1 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @collective_assign_group_v2
func @collective_assign_group_v2(%input: tensor<f32>) -> tensor<f32> {
  %rank = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %group_assignment = "tf.Const"() { value = dense<[[0, 1]]> : tensor<1x2xi32> } : () -> tensor<1x2xi32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %group_key = "tf.CollectiveAssignGroupV2"(%group_assignment, %rank) {} : (tensor<1x2xi32>, tensor<i32>) -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: channel_handle = {handle = 1 : i64, type = 1 : i64}
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// -----

func @inconsistent_collective_info(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<11> : tensor<i32> } : () -> tensor<i32>
  %group_size1 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size2 = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.CollectiveReduceV2"(%input, %group_size1, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  // expected-error@below {{op module already contains an attribute tf2xla.collective_info.group_size=1, overwritting to a new value 2 is not allowed.}}
  %1 = "tf.CollectiveReduceV2"(%input, %group_size2, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}
