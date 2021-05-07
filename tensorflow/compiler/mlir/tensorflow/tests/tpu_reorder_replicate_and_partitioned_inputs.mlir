// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-reorder-replicate-partitioned-inputs | FileCheck %s

// CHECK-LABEL:func @simple
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>)
func @simple(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  // CHECK: [[RI_0:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG2]])
  // CHECK: [[RI_1:%.*]] = "tf.TPUReplicatedInput"([[ARG1]], [[ARG3]])
  // CHECK: [[PI:%.*]] = "tf.TPUPartitionedInput"([[RI_0]], [[RI_1]])
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // CHECK: return [[PI]]
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// CHECK-LABEL:func @missing_xla_sharding
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>)
func @missing_xla_sharding(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  // CHECK: [[RI_0:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG2]])
  // CHECK: [[RI_1:%.*]] = "tf.TPUReplicatedInput"([[ARG1]], [[ARG3]])
  // CHECK: [[PI:%.*]] = "tf.TPUPartitionedInput"([[RI_0]], [[RI_1]])
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {device = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {device = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // CHECK: return [[PI]]
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// Test IR is not modified when none of the operands of tf.TPUReplicaedInput is
// a tf.TPUPartitionedInput op.

// CHECK-LABEL:func @no_change_to_dag
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf.resource<tensor<10x3xf32>>>)
func @no_change_to_dag(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInput"([[ARG0]], [[ARG1]])
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {device = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // CHECK: [[PI_1:%.*]] = "tf.TPUPartitionedInput"([[ARG2]], [[ARG3]])
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {device = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG1]])
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // CHECK: return [[RI]], [[PI_0]], [[PI_1]]
  return %ri, %pi_0, %pi_1 : tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>
}

// -----

func @xla_sharding_mismatch(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "123", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects all inputs from 'tf.TPUPartitionedInput' ops to have identical XLA sharding}}
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// -----

func @partition_dim_mismatch(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects partition_dim = -1 but found 0}}
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "", partition_dim = 0 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// -----

func @num_partitioned_inputs_mismatch(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>, %arg4: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects 2 operands but found 3}}
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3, %arg4) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// -----

func @unsupported_replicated_input_index(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{'tf.TPUReplicatedInput' op unsupported index = 1}}
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) {index = 1} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}

// -----

func @mixed_inputs_to_replicated_op(%arg0: tensor<!tf.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{'tf.TPUReplicatedInput' op expects all inputs from 'tf.TPUPartitionedInput' ops}}
  %ri = "tf.TPUReplicatedInput"(%pi_0, %arg2) {index = 1} : (tensor<!tf.resource<tensor<10x3xf32>>>, tensor<!tf.resource<tensor<10x3xf32>>>) -> tensor<!tf.resource<tensor<10x3xf32>>>
  return %ri : tensor<!tf.resource<tensor<10x3xf32>>>
}
