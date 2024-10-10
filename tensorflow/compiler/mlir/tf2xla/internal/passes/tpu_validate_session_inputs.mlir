// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-validate-session-inputs | FileCheck %s

// CHECK-LABEL: func @does_not_contian_InfeedDequeueTuple
func.func @does_not_contian_InfeedDequeueTuple(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
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
func.func @contians_InfeedDequeueTuple(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island() wraps "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
    %ri, %c0 = tf_executor.island wraps "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %out, %c1 = tf_executor.island wraps "tf.opA"(%ri) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
    // expected-warning @+1 {{TPU_REPLICARTED_CORE:0 device is not supported for op = tf.InfeedDequeueTuple in TF2XLA MLIR Bridge}}
    %infeed_output:3, %c2 = tf_executor.island wraps "tf.InfeedDequeueTuple"() {device = "/device:TPU_REPLICATED_CORE:0"} : () -> (tensor<3xi32>, tensor<4x?xf32>, tensor<*xi16>)
    %ro:2, %c3 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%out) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    tf_executor.fetch %ro#0, %ro#1 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}