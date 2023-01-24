// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-reorder-replicate-partitioned-inputs | FileCheck %s

// CHECK-LABEL:func @simple
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
func.func @simple(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  // CHECK: [[RI_0:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG2]])
  // CHECK: [[RI_1:%.*]] = "tf.TPUReplicatedInput"([[ARG1]], [[ARG3]])
  // CHECK: [[PI:%.*]] = "tf.TPUPartitionedInputV2"([[RI_0]], [[RI_1]])
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: return [[PI]]
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// CHECK-LABEL:func @missing_xla_sharding
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
func.func @missing_xla_sharding(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  // CHECK: [[RI_0:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG2]])
  // CHECK: [[RI_1:%.*]] = "tf.TPUReplicatedInput"([[ARG1]], [[ARG3]])
  // CHECK: [[PI:%.*]] = "tf.TPUPartitionedInputV2"([[RI_0]], [[RI_1]])
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {device = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3) {device = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: return [[PI]]
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// Test IR is not modified when none of the operands of tf.TPUReplicaedInput is
// a tf.TPUPartitionedInputV2 op.

// CHECK-LABEL:func @no_change_to_dag
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
func.func @no_change_to_dag(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInputV2"([[ARG0]], [[ARG1]])
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {device = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: [[PI_1:%.*]] = "tf.TPUPartitionedInputV2"([[ARG2]], [[ARG3]])
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3) {device = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[ARG0]], [[ARG1]])
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: return [[RI]], [[PI_0]], [[PI_1]]
  func.return %ri, %pi_0, %pi_1 : tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// -----

func.func @xla_sharding_mismatch(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3) {_XlaSharding = "123", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects all inputs from 'tf.TPUPartitionedInputV2' ops to have identical XLA sharding}}
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// -----

func.func @partition_dim_mismatch(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects partition_dims = [] but found [1, 2]}}
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3) {_XlaSharding = "", partition_dims = [1, 2]} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// -----

func.func @num_partitioned_inputs_mismatch(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg4: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{expects 2 operands but found 3}}
  %pi_1 = "tf.TPUPartitionedInputV2"(%arg2, %arg3, %arg4) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// -----

func.func @mixed_inputs_to_replicated_op(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  %pi_0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {_XlaSharding = "", partition_dims = []} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // expected-error@+1 {{'tf.TPUReplicatedInput' op expects all inputs from 'tf.TPUPartitionedInputV2' ops}}
  %ri = "tf.TPUReplicatedInput"(%pi_0, %arg2) {index = 1} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}
