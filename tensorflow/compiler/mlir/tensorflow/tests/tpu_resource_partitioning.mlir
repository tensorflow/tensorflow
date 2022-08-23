// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-resource-partition | FileCheck %s

func.func private @computation(%arg0: tensor<i32>) -> tensor<i32>

// CHECK-LABEL: func @read_write_resource
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func @read_write_resource(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-DAG:  [[READ0:%.+]] = "tf.ReadVariableOp"([[ARG0]])
  // CHECK-DAG:  [[READ1:%.+]] = "tf.ReadVariableOp"([[ARG1]])
  // CHECK:      [[INPUT:%.+]] = "tf.TPUPartitionedInput"([[READ0]], [[READ1]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dim = -1
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK:      [[COMPUTATION:%.+]] = "tf_device.cluster_func"([[INPUT]])
  %2 = "tf_device.cluster_func"(%1) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
  // CHECK:      [[OUTPUT:%.+]]:2 = "tf.TPUPartitionedOutput"([[COMPUTATION]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dim = -1
  // CHECK-DAG:  "tf.AssignVariableOp"([[ARG0]], [[OUTPUT]]#0)
  // CHECK-DAG:  "tf.AssignVariableOp"([[ARG1]], [[OUTPUT]]#1)
  "tf.AssignVariableOp"(%0, %2) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @read_only_resource
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func @read_only_resource(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK-DAG:  [[READ0:%.+]] = "tf.ReadVariableOp"([[ARG0]])
  // CHECK-DAG:  [[READ1:%.+]] = "tf.ReadVariableOp"([[ARG1]])
  // CHECK:      [[INPUT:%.+]] = "tf.TPUPartitionedInput"([[READ0]], [[READ1]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dim = -1
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK:      "tf_device.cluster_func"([[INPUT]])
  %2 = "tf_device.cluster_func"(%1) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT:  tf.TPUPartitionedOutput
  // CHECK-NOT:  tf.AssignVariableOp
  func.return %2 : tensor<i32>
}

func.func private @computation_two_args(%arg0: tensor<i32>, %arg1: tensor<i32>)

// CHECK-LABEL: func @partitioned_variable_multiple_users
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func @partitioned_variable_multiple_users(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-DAG:  [[READ0:%.+]] = "tf.ReadVariableOp"([[ARG0]])
  // CHECK-DAG:  [[READ1:%.+]] = "tf.ReadVariableOp"([[ARG1]])
  // CHECK:      [[INPUT0:%.+]] = "tf.TPUPartitionedInput"([[READ0]], [[READ1]])
  // CHECK-DAG:  [[READ2:%.+]] = "tf.ReadVariableOp"([[ARG0]])
  // CHECK-DAG:  [[READ3:%.+]] = "tf.ReadVariableOp"([[ARG1]])
  // CHECK:      [[INPUT1:%.+]] = "tf.TPUPartitionedInput"([[READ2]], [[READ3]])
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK:      "tf_device.cluster_func"([[INPUT0]], [[INPUT1]])
  "tf_device.cluster_func"(%1, %2) {func = @computation_two_args, use_spmd_for_xla_partitioning = true} : (tensor<i32>, tensor<i32>) -> ()
  func.return
}

// Tests unsupported cases and IR are not modified.

// CHECK-LABEL: func @no_spmd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func @no_spmd(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK:      "tf.TPUPartitionedInput"([[ARG0]], [[ARG1]])
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf_device.cluster_func"(%1) {func = @computation} : (tensor<i32>) -> tensor<i32>
  // CHECK:      "tf.TPUPartitionedInput"([[ARG0]], [[ARG1]])
  %3 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %4 = "tf.ReadVariableOp"(%3) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %5 = "tf_device.cluster_func"(%4) {func = @computation, use_spmd_for_xla_partitioning = false} : (tensor<i32>) -> tensor<i32>
  func.return
}

// CHECK-LABEL: func @read_write_unpartitioned_resource
func.func @read_write_unpartitioned_resource(%arg0: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-NOT:  tf.TPUPartitionedInput
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1 = "tf_device.cluster_func"(%0) {func = @computation} : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT:  tf.TPUPartitionedOutput
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @read_only_unpartitioned_resource
func.func @read_only_unpartitioned_resource(%arg0: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-NOT:  tf.TPUPartitionedInput
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1 = "tf_device.cluster_func"(%0) {func = @computation} : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT:  tf.TPUPartitionedOutput
  // CHECK-NOT:  tf.AssignVariableOp
  func.return
}

// CHECK-LABEL: func @resource_read_multiple_users
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
func.func @resource_read_multiple_users(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK:      "tf.TPUPartitionedInput"([[ARG0]], [[ARG1]])
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf_device.cluster_func"(%1) {func = @computation} : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @non_resource_read_input_write_output
func.func @non_resource_read_input_write_output(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  tf.TPUPartitionedInput
  %0 = "tf_device.cluster_func"(%arg0) {func = @computation} : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT:  tf.TPUPartitionedOutput
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @resource_missing_subtype
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource>, [[ARG1:%.+]]: tensor<!tf_type.resource>)
func.func @resource_missing_subtype(%arg0: tensor<!tf_type.resource>, %arg1: tensor<!tf_type.resource>) {
  // CHECK:      "tf.TPUPartitionedInput"([[ARG0]], [[ARG1]])
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<!tf_type.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource>) -> tensor<i32>
  %2 = "tf_device.cluster_func"(%1) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT:  tf.TPUPartitionedOutput
  "tf.AssignVariableOp"(%0, %2) : (tensor<!tf_type.resource>, tensor<i32>) -> ()
  func.return
}

// -----

// Check outside compiled that uses a TPUPartitionedInput.

func.func private @computation(%arg0: tensor<i32>) -> tensor<i32>

// CHECK-LABEL: func @with_host_process
// CHECK-SAME: ([[ARG0:%.+]]: tensor<!tf_type.resource<tensor<i32>>>, [[ARG1:%.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func @with_host_process(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-DAG:  [[READ0:%.+]] = "tf.ReadVariableOp"([[ARG0]])
  // CHECK-DAG:  [[READ1:%.+]] = "tf.ReadVariableOp"([[ARG1]])
  // CHECK:      [[INPUT:%.+]] = "tf.TPUPartitionedInput"([[READ0]], [[READ1]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dim = -1
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK:      [[COMPUTATION:%.+]] = "tf_device.parallel_execute"()
  // CHECK:      "tf.OpA"([[READ0]])
  %2 = "tf_device.parallel_execute"() ({
    "tf_device.launch"() ({
      "tf.OpA"(%1) : (tensor<i32>) -> ()
      tf_device.return
    }) {device = "TPU_REPLICATED_HOST"} : () -> ()
    tf_device.return
  }, {
    %3 = "tf_device.cluster_func"(%1) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) : () -> tensor<i32>
  // CHECK:      [[OUTPUT:%.+]]:2 = "tf.TPUPartitionedOutput"([[COMPUTATION]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dim = -1
  // CHECK-DAG:  "tf.AssignVariableOp"([[ARG0]], [[OUTPUT]]#0)
  // CHECK-DAG:  "tf.AssignVariableOp"([[ARG1]], [[OUTPUT]]#1)
  "tf.AssignVariableOp"(%0, %2) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}

// -----

// Check for an error that reports the unsupported case of outside compiled
// code that uses a TPUPartitionedInput without REPLICATED sharding.

// The TPUParitionedInput has the following OpSharding:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"

func.func private @computation(%arg0: tensor<i32>) -> tensor<i32>

func.func @non_replicated_sharding(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
  // expected-error@+1 {{support}}
  %0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "\08\03\1A\02\01\02\22\02\00\01", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf_device.parallel_execute"() ({
    "tf_device.launch"() ({
      "tf.OpA"(%1) : (tensor<i32>) -> ()
      tf_device.return
    }) {device = "TPU_REPLICATED_HOST"} : () -> ()
    tf_device.return
  }, {
    %3 = "tf_device.cluster_func"(%1) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) : () -> tensor<i32>
  "tf.AssignVariableOp"(%0, %2) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}
