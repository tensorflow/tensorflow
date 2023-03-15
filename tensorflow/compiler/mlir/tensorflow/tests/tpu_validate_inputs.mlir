// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-validate-inputs | FileCheck %s

// CHECK-LABEL: func @num_replicas_replicated
func.func @num_replicas_replicated(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_input(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUReplicatedInput' op TF/XLA TPU bridge input check: number of inputs inconsistent. num_replicas=2 no. of inputs=3}}
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_input_packed(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUReplicatedInput' op TF/XLA TPU bridge input check: packed with number of inputs not 1. num_replicas=2 no. of inputs=2}}
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = true} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_replicas_replicated_output(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.TPUReplicatedOutput' op TF/XLA TPU bridge input check: number of outputs inconsistent. num_replicas=2 no. of outputs=3}}
    %ro:3, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_core_per_replica_partitioned_input(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUPartitionedInput' op TF/XLA TPU bridge input check: number of inputs inconsistent. num_cores_per_replica=2 no. of inputs=3}}
    %pi, %c0 = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1, %arg1) {index = 1 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%pi) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %po:2, %c2 = tf_executor.island wraps "tf.TPUPartitionedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %po#0, %po#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_core_per_replica_partitioned_output(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    %pi, %c0 = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {index = 1 : i64} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%pi) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.TPUPartitionedOutput' op TF/XLA TPU bridge input check: number of outputs inconsistent. num_cores_per_replica=2 no. of outputs=3}}
    %po:3, %c2 = tf_executor.island wraps "tf.TPUPartitionedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    tf_executor.fetch %po#0, %po#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @validate_tpu_replicate_no_attr(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.opA' op TF/XLA TPU bridge input check: tf.TPUReplicatedInput op has successor op tf.opA with error: missing _tpu_replicate attr}}
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.opB' op TF/XLA TPU bridge input check: tf.TPUReplicatedOutput op has predecessor op tf.opB with error: missing _tpu_replicate attr}}
    %out2, %c2 = tf_executor.island wraps "tf.opB"(%out) : (tensor<i32>) -> tensor<i32>
    %ro:2, %c3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out2) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @validate_tpu_replicate_wrong_attr(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.opA' op TF/XLA TPU bridge input check: tf.TPUReplicatedInput op has successor op tf.opA with error: invalid _tpu_replicate attr. Expected attr: "cluster", Actual attr: "cluster_wrong"}}
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster_wrong"}: (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.opB' op TF/XLA TPU bridge input check: tf.TPUReplicatedOutput op has predecessor op tf.opB with error: invalid _tpu_replicate attr. Expected attr: "cluster", Actual attr: "cluster_wrong"}}
    %out2, %c2 = tf_executor.island wraps "tf.opB"(%out) {_tpu_replicate = "cluster_wrong"}: (tensor<i32>) -> tensor<i32>
    %ro:2, %c3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out2) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----
