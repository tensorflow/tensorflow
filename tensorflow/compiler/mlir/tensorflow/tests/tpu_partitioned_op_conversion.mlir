// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-partitioned-op-conversion | FileCheck %s

// CHECK-LABEL:func @replicated
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
func.func @replicated(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>> {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInputV2"([[ARG0]], [[ARG1]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dims = []
  // CHECK: [[PI_1:%.*]] = "tf.TPUPartitionedInputV2"([[ARG2]], [[ARG3]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK-SAME: partition_dims = []
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[PI_0]], [[PI_1]])
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<10x3xf32>>>
  // CHECK: return [[RI]]
  func.return %ri : tensor<!tf_type.resource<tensor<10x3xf32>>>
}

// -----

// CHECK-LABEL:func @partitioned_2d
// CHECK-SAME: ([[ARG0:%.*]]: tensor<10x3xf32>, [[ARG1:%.*]]: tensor<10x3xf32>)
func.func @partitioned_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<10x3xf32>) -> tensor<20x3xf32> {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInputV2"([[ARG0]], [[ARG1]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [2, 1]
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[PI_0]])
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 1 : i64} : () -> ()
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "123", partition_dim = 0 : i64} : (tensor<10x3xf32>, tensor<10x3xf32>) -> tensor<20x3xf32>
  %ri = "tf.TPUReplicatedInput"(%pi_0) : (tensor<20x3xf32>) -> tensor<20x3xf32>
  // CHECK: return [[RI]]
  func.return %ri : tensor<20x3xf32>
}

// -----

// CHECK-LABEL:func @partitioned_2d_resource
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
func.func @partitioned_2d_resource(%arg0: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg2: tensor<!tf_type.resource<tensor<10x3xf32>>>, %arg3: tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<20x3xf32>>> {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInputV2"([[ARG0]], [[ARG1]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [2, 1]
  // CHECK: [[PI_1:%.*]] = "tf.TPUPartitionedInputV2"([[ARG2]], [[ARG3]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [2, 1]
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[PI_0]], [[PI_1]])
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 2 : i64} : () -> ()
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "123", partition_dim = 0 : i64} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<20x3xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "123", partition_dim = 0 : i64} : (tensor<!tf_type.resource<tensor<10x3xf32>>>, tensor<!tf_type.resource<tensor<10x3xf32>>>) -> tensor<!tf_type.resource<tensor<20x3xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<20x3xf32>>>, tensor<!tf_type.resource<tensor<20x3xf32>>>) -> tensor<!tf_type.resource<tensor<20x3xf32>>>
  // CHECK: return [[RI]]
  func.return %ri : tensor<!tf_type.resource<tensor<20x3xf32>>>
}

// -----

// CHECK-LABEL:func @partitioned_3d
// CHECK-SAME: ([[ARG0:%.*]]: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, [[ARG1:%.*]]: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, [[ARG2:%.*]]: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, [[ARG3:%.*]]: tensor<!tf_type.resource<tensor<16x8x16xf32>>>)
func.func @partitioned_3d(%arg0: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, %arg1: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, %arg2: tensor<!tf_type.resource<tensor<16x8x16xf32>>>, %arg3: tensor<!tf_type.resource<tensor<16x8x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x16x16xf32>>> {
  // CHECK: [[PI_0:%.*]] = "tf.TPUPartitionedInputV2"([[ARG0]], [[ARG1]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [1, 2, 1]
  // CHECK: [[PI_1:%.*]] = "tf.TPUPartitionedInputV2"([[ARG2]], [[ARG3]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [1, 2, 1]
  // CHECK: [[RI:%.*]] = "tf.TPUReplicatedInput"([[PI_0]], [[PI_1]])
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 2 : i64} : () -> ()
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "123", partition_dim = 1 : i64} : (tensor<!tf_type.resource<tensor<16x8x16xf32>>>, tensor<!tf_type.resource<tensor<16x8x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x16x16xf32>>>
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "123", partition_dim = 1 : i64} : (tensor<!tf_type.resource<tensor<16x8x16xf32>>>, tensor<!tf_type.resource<tensor<16x8x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x16x16xf32>>>
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (tensor<!tf_type.resource<tensor<16x16x16xf32>>>, tensor<!tf_type.resource<tensor<16x16x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x16x16xf32>>>
  // CHECK: return [[RI]]
  func.return %ri : tensor<!tf_type.resource<tensor<16x16x16xf32>>>
}

// -----

// CHECK-LABEL:func @partitioned_output_3d
// CHECK-SAME: ([[ARG:%.*]]: tensor<!tf_type.resource<tensor<16x16x16xf32>>>)
func.func @partitioned_output_3d(%arg: tensor<!tf_type.resource<tensor<16x16x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x8x16xf32>>> {
  // CHECK: [[PO:%.*]] = "tf.TPUPartitionedOutputV2"([[ARG]])
  // CHECK-SAME: _XlaSharding = "123"
  // CHECK-SAME: partition_dims = [1, 2, 1]
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 2 : i64} : () -> ()
  %po:2 = "tf.TPUPartitionedOutput"(%arg) {_XlaSharding = "123", partition_dim = 1 : i64} : (tensor<!tf_type.resource<tensor<16x16x16xf32>>>) -> (tensor<!tf_type.resource<tensor<16x8x16xf32>>>, tensor<!tf_type.resource<tensor<16x8x16xf32>>>)
  // CHECK: return [[PO:%.*0]]
  func.return %po#0 : tensor<!tf_type.resource<tensor<16x8x16xf32>>>
}

// -----

func.func @out_of_range_dim(%arg: tensor<!tf_type.resource<tensor<16x16x16xf32>>>) -> tensor<!tf_type.resource<tensor<16x8x16xf32>>> {
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 2 : i64} : () -> ()
  // expected-error @+1 {{cannot partition 'tensor<16x16x16xf32>' (rank = 3) along dimension 3.}}
  %po:2 = "tf.TPUPartitionedOutput"(%arg) {_XlaSharding = "123", partition_dim = 3 : i64} : (tensor<!tf_type.resource<tensor<16x16x16xf32>>>) -> (tensor<!tf_type.resource<tensor<16x8x16xf32>>>, tensor<!tf_type.resource<tensor<16x8x16xf32>>>)
  func.return %po#0 : tensor<!tf_type.resource<tensor<16x8x16xf32>>>
}

// -----

func.func @unranked(%arg: tensor<!tf_type.resource<tensor<*xf32>>>) -> tensor<!tf_type.resource<tensor<*xf32>>> {
  "tf.TPUReplicateMetadata"() {num_cores_per_replica = 2 : i64, num_replicas = 2 : i64} : () -> ()
  // expected-error @+1 {{cannot convert op with unranked or non-tensor input type 'tensor<*xf32>'.}}
  %po:2 = "tf.TPUPartitionedOutput"(%arg) {_XlaSharding = "123", partition_dim = 3 : i64} : (tensor<!tf_type.resource<tensor<*xf32>>>) -> (tensor<!tf_type.resource<tensor<*xf32>>>, tensor<!tf_type.resource<tensor<*xf32>>>)
  func.return %po#0 : tensor<!tf_type.resource<tensor<*xf32>>>
}
