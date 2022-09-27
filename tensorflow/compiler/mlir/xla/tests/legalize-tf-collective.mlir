// RUN: xla-opt -split-input-file -verify-diagnostics -xla-legalize-tf-collective -xla-legalize-tf=allow-partial-conversion %s | FileCheck %s


// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @all_reduce_cross_replica
func.func @all_reduce_cross_replica(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  // CHECK-NOT: channel_handle
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @all_reduce_cross_replica_and_partition
func.func @all_reduce_cross_replica_and_partition(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT: channel_handle = #mhlo.channel_handle<handle = 2, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT: channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  %1 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplicaAndPartition"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// CHECK-LABEL: func @xla_all_reduce_add
func.func @xla_all_reduce_add(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Add", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @xla_all_reduce_max
func.func @xla_all_reduce_max(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.maximum
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Max", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @xla_all_reduce_mean
func.func @xla_all_reduce_mean(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: %[[GROUP_SIZE:.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: %[[RESULT:.*]] = mhlo.divide %[[REDUCE]], %[[GROUP_SIZE]]
  // CHECK-NEXT: return %[[RESULT]]
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Mean", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @xla_all_reduce_min
func.func @xla_all_reduce_min(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.minimum
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Min", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @xla_all_reduce_mul
func.func @xla_all_reduce_mul(%input: tensor<f32>) -> tensor<f32> {
  %group_assignment = "tf.Const"() { value = dense<[[0],[1]]> : tensor<2x1xi32> } : () -> tensor<2x1xi32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.mul
  %0 = "tf.XlaAllReduce"(%input, %group_assignment) {reduce_op = "Mul", mode = "CrossReplica"} : (tensor<f32>, tensor<2x1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}


// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 1
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @collective_reduce_v2
func.func @collective_reduce_v2(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT: channel_handle = #mhlo.channel_handle<handle = 2, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT: channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  %1 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// CHECK-LABEL: func @collective_reduce_v2_add_id
func.func @collective_reduce_v2_add_id(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: return %[[REDUCE]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_max_id
func.func @collective_reduce_v2_max_id(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: return %[[REDUCE]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Max", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_min_id
func.func @collective_reduce_v2_min_id(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.minimum
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: return %[[REDUCE]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Min", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_mul_id
func.func @collective_reduce_v2_mul_id(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.mul
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: return %[[REDUCE]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Mul", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_add_div
func.func @collective_reduce_v2_add_div(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[GROUP_SIZE:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = mhlo.divide %[[REDUCE]], %[[GROUP_SIZE]]
  // CHECK-NEXT: return %[[RESULT]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Div"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_max_div
func.func @collective_reduce_v2_max_div(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[GROUP_SIZE:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = mhlo.divide %[[REDUCE]], %[[GROUP_SIZE]]
  // CHECK-NEXT: return %[[RESULT]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Max", final_op = "Div"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_min_div
func.func @collective_reduce_v2_min_div(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[GROUP_SIZE:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.minimum
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = mhlo.divide %[[REDUCE]], %[[GROUP_SIZE]]
  // CHECK-NEXT: return %[[RESULT]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Min", final_op = "Div"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @collective_reduce_v2_mul_div
func.func @collective_reduce_v2_mul_div(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // CHECK: %[[GROUP_SIZE:.*]] = mhlo.constant dense<2.000000e+00>
  // CHECK: %[[REDUCE:.*]] = "mhlo.all_reduce"
  // CHECK: mhlo.mul
  // CHECK: mhlo.return
  // CHECK-NEXT{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = mhlo.divide %[[REDUCE]], %[[GROUP_SIZE]]
  // CHECK-NEXT: return %[[RESULT]]
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Mul", final_op = "Div"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}


// -----

// CHECK: module attributes
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_key = 0
// CHECK-SAME{LITERAL}: tf2xla.collective_info.group_size = 2
// CHECK-LABEL: func @collective_assign_group_v2
func.func @collective_assign_group_v2(%input: tensor<f32>) -> tensor<f32> {
  %rank = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %key_base = "tf.Const"() { value = dense<10> : tensor<i32> } : () -> tensor<i32>
  %group_assignment = "tf.Const"() { value = dense<[[0, 1]]> : tensor<1x2xi32> } : () -> tensor<1x2xi32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %group_size, %group_key = "tf.CollectiveAssignGroupV2"(%group_assignment, %rank, %key_base) {} : (tensor<1x2xi32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  // CHECK-NOT: "tf.CollectiveAssignGroupV2"
  // CHECK: "mhlo.all_reduce"
  // CHECK: mhlo.add
  // CHECK{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-NOT: "tf.CollectiveAssignGroupV2"
  %0 = "tf.CollectiveReduceV2"(%input, %group_size, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @inconsistent_collective_info(%input: tensor<f32>) -> tensor<f32> {
  %group_key = "tf.Const"() { value = dense<11> : tensor<i32> } : () -> tensor<i32>
  %group_size1 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %group_size2 = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %instance_key = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  // expected-error@below {{op module already contains an attribute tf2xla.collective_info.group_size=2, overwritting to a new value 1 is not allowed.}}
  %0 = "tf.CollectiveReduceV2"(%input, %group_size1, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %1 = "tf.CollectiveReduceV2"(%input, %group_size2, %group_key, %instance_key) {merge_op = "Add", final_op = "Id"} : (tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

