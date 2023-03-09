// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-validate-inputs | FileCheck %s

// CHECK-LABEL: func @num_replicas_replicated
func.func @num_replicas_replicated(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %out = "tf.opA"(%ri) : (tensor<i32>) -> tensor<i32>
  %ro:2 = "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %ro#0, %ro#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_input(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  // expected-error @+1 {{'tf.TPUReplicatedInput' op TF/XLA TPU bridge input check: number of inputs inconsistent. num_replicas=2 no. of inputs=3}}
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %ro:2 = "tf.TPUReplicatedOutput"(%ri) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %ro#0, %ro#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_input_packed(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  // expected-error @+1 {{'tf.TPUReplicatedInput' op TF/XLA TPU bridge input check: packed with number of inputs not 1. num_replicas=2 no. of inputs=2}}
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = true} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %ro:2 = "tf.TPUReplicatedOutput"(%ri) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %ro#0, %ro#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_output(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // expected-error @+1 {{'tf.TPUReplicatedOutput' op TF/XLA TPU bridge input check: number of outputs inconsistent. num_replicas=2 no. of outputs=3}}
  %ro:3 = "tf.TPUReplicatedOutput"(%ri) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  func.return %ro#0, %ro#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_core_per_replica_partitioned_input(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
  // expected-error @+1 {{'tf.TPUPartitionedInput' op TF/XLA TPU bridge input check: number of inputs inconsistent. num_cores_per_replica=2 no. of inputs=3}}
  %pi = "tf.TPUPartitionedInput"(%arg0, %arg1, %arg1) {index = 1 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %po:2 = "tf.TPUPartitionedOutput"(%pi) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %po#0, %po#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_core_per_replica_partitioned_output(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
  %pi = "tf.TPUPartitionedInput"(%arg0, %arg1) {index = 1 : i64} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // expected-error @+1 {{'tf.TPUPartitionedOutput' op TF/XLA TPU bridge input check: number of outputs inconsistent. num_cores_per_replica=2 no. of outputs=3}}
  %po:3 = "tf.TPUPartitionedOutput"(%pi) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  func.return %po#0, %po#1 : tensor<i32>, tensor<i32>
}