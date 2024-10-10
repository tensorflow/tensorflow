// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-validate-inputs | FileCheck %s

// CHECK-LABEL: func @num_replicas_replicated
func.func @num_replicas_replicated(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
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
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUReplicatedInput' op TF2XLA TPU bridge input check: number of inputs inconsistent. num_replicas=2 no. of inputs=3}}
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
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUReplicatedInput' op TF2XLA TPU bridge input check: packed with number of inputs not 1. num_replicas=2 no. of inputs=2}}
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
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.TPUReplicatedOutput' op TF2XLA TPU bridge input check: number of outputs inconsistent. num_replicas=2 no. of outputs=3}}
    %ro:3, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @num_core_per_replica_partitioned_input(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.TPUPartitionedInput' op TF2XLA TPU bridge input check: number of inputs inconsistent. num_cores_per_replica=2 no. of inputs=3}}
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
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    %pi, %c0 = tf_executor.island wraps "tf.TPUPartitionedInput"(%arg0, %arg1) {index = 1 : i64} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%pi) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.TPUPartitionedOutput' op TF2XLA TPU bridge input check: number of outputs inconsistent. num_cores_per_replica=2 no. of outputs=3}}
    %po:3, %c2 = tf_executor.island wraps "tf.TPUPartitionedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    tf_executor.fetch %po#0, %po#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @validate_tpu_replicate_no_attr(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate="cluster"}: (tensor<i32>) -> tensor<i32>
    // expected-warning @+1 {{TF2XLA TPU bridge input check: cluster op = tf.opA with cluster = cluster has successor as non cluster op tf.opB}}
    %out2, %c2 = tf_executor.island wraps "tf.opB"(%out) : (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{tf.TPUReplicatedOutput' op TF2XLA TPU bridge input check: non-cluster op = tf.opB has invalid successor op = tf.TPUReplicatedOutput}}
    %ro:2, %c4 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out2) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @validate_tpu_replicate_wrong_attr(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster_wrong"}: (tensor<i32>) -> tensor<i32>
    // expected-error @+1 {{'tf.opB' op TF2XLA TPU bridge input check: mismatch clusters tpu_replicate attr. Parent op tf.opA with cluster = cluster_wrong has successor cluster op tf.opB with cluster = cluster}}
    %out2, %c2 = tf_executor.island wraps "tf.opB"(%out) {_tpu_replicate = "cluster"}: (tensor<i32>) -> tensor<i32>
    %ro:2, %c3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out2) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @valid_xla_nonxla(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster", device = "TPU"} : (tensor<i32>) -> tensor<i32>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

func.func @valid_xla_nonxla_warning(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<*x!tf_type.string>, tensor<*x!tf_type.string>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<*x!tf_type.string>
    // expected-warning @+1 {{TF/XLA TPU bridge input check: found invalid op. tf.Identity can't be both xla and non-xla}}
    %out, %c1 = tf_executor.island(%c0) wraps "tf.Identity"(%ri) {_tpu_replicate = "cluster", device = ""} : (tensor<*x!tf_type.string>) -> tensor<*x!tf_type.string>
    %ro:2, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<*x!tf_type.string>) -> (tensor<*x!tf_type.string>, tensor<*x!tf_type.string>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<*x!tf_type.string>, tensor<*x!tf_type.string>
  }
  return %0#0, %0#1 : tensor<*x!tf_type.string>, tensor<*x!tf_type.string>
}

// -----

//  Serialized string:
//   "\08\01\1A\01\01\22\01\00"
//  Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0

func.func @valid_MAXIMAL_sharding_device(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    %0, %c = tf_executor.island wraps "tf.Identity"(%arg0) {_tpu_replicate = "cluster", _XlaSharding = "\08\01\1A\01\01\22\01\00"} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %0 : tensor<i32>
  }
  return %0 : tensor<i32>
}
// -----

//  Serialized string:
//   "\08\01\1A\01\01\22\01\02"
//  Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 2

func.func @invalid_MAXIMAL_sharding_device(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.Identity' op TF2XLA TPU bridge input check: invalid sharding device 2 for num_cores_per_replica = 2}}
    %0, %c = tf_executor.island wraps "tf.Identity"(%arg0) {_tpu_replicate = "cluster", _XlaSharding = "\08\01\1A\01\01\22\01\02"} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %0 : tensor<i32>
  }
  return %0 : tensor<i32>
}
// -----

//  Serialized string:
//   "\08\02\2a\08\08\01\1a\01\01\22\01\00\2a\08\08\01\1a\01\01\22\01\01"
//  Proto debug string:
//    type: 	 TUPLE
//    tuple_shardings {
//      type: MAXIMAL
//      tile_assignment_dimensions: 1
//      tile_assignment_devices: 0
//    }
//    tuple_shardings {
//      type: MAXIMAL
//      tile_assignment_dimensions: 1
//      tile_assignment_devices: 1
//    }

func.func @invalid_TUPLE_sharding_arity(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    // expected-error @+1 {{'tf.Identity' op TF2XLA TPU bridge input check: invalid no. of tuple shardings 2 for arity = 1}}
    %0, %c = tf_executor.island wraps "tf.Identity"(%arg0) {_tpu_replicate = "cluster", _XlaSharding = "\08\02\2a\08\08\01\1a\01\01\22\01\00\2a\08\08\01\1a\01\01\22\01\01"} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %0 : tensor<i32>
  }
  return %0 : tensor<i32>
}
// -----

//  Serialized string:
//   "\08\02\2a\08\08\01\1a\01\01\22\01\00\2a\08\08\01\1a\01\01\22\01\01"
//  Proto debug string:
//    type: 	 TUPLE
//    tuple_shardings {
//      type: MAXIMAL
//      tile_assignment_dimensions: 1
//      tile_assignment_devices: 0
//    }
//    tuple_shardings {
//      type: MAXIMAL
//      tile_assignment_dimensions: 1
//      tile_assignment_devices: 1
//    }

func.func @outfeed_enqueue_tuple_sharding_exception(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %control = tf_executor.island wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 1 : i64, topology = "topology"} : () -> ()
    %0, %c0 = tf_executor.island wraps "tf.AddV2"(%arg0, %arg1) {_tpu_replicate = "cluster"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c1 = tf_executor.island wraps "tf.OutfeedEnqueueTuple"(%arg0, %arg1) {_tpu_replicate = "cluster", _XlaSharding = "\08\02\2a\08\08\01\1a\01\01\22\01\00\2a\08\08\01\1a\01\01\22\01\01"} : (tensor<i32>, tensor<i32>) -> ()
    tf_executor.fetch %0 : tensor<i32>
  }
  return %0 : tensor<i32>
}


// -----

func.func @single_core_tpu(%arg0: tensor<i32>) -> () {
  tf_executor.graph {
    // expected-error @+1 {{found a single-core TPU graph}}
    tf_executor.island wraps "tf.Identity"(%arg0) {_xla_compile_device_type = "TPU"} : (tensor<i32>) -> tensor<i32>
     tf_executor.fetch
  }
  return
}

// -----

// CHECK-LABEL: func @num_replicas_1
func.func @num_replicas_1(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 1, num_cores_per_replica = 1, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    %ro, %c2 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %ro : tensor<i32>
  }
  return %0 : tensor<i32>
}
