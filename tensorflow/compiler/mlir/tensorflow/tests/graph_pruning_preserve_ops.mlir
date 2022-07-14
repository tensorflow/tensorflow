// RUN: tf-opt %s -split-input-file -tf-executor-graph-pruning=ops-to-preserve="tf.TPUReplicateMetadata,tf.TPUCompilationResult,tf.TPUReplicatedInput,tf.TPUReplicatedOutput,tf.CustomOp" | FileCheck %s

// Verifies that specified ops, and ops reachable from those, are preserved.

// CHECK-LABEL: func @preserve_unreachable_tpu_replicate_metadata
func.func @preserve_unreachable_tpu_replicate_metadata() {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    // CHECK: "tf.NoOp"
    %1 = tf_executor.island wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.TPUReplicateMetadata"
    %2 = tf_executor.island(%1) wraps "tf.TPUReplicateMetadata"() {allow_soft_placement = false, computation_shape = [], device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
    tf_executor.fetch %0 : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @preserve_unreachable_tpu_compilation_result
func.func @preserve_unreachable_tpu_compilation_result() {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    // CHECK: "tf.NoOp"
    %1 = tf_executor.island wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.TPUCompilationResult"
    %2, %3 = tf_executor.island(%1) wraps "tf.TPUCompilationResult"() : () -> tensor<!tf_type.string>
    tf_executor.fetch %0 : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @preserve_unreachable_tpu_replicated_input
func.func @preserve_unreachable_tpu_replicated_input(%arg0: tensor<i1>) {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    // CHECK: "tf.NoOp"
    %1 = tf_executor.island wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"
    %2, %3 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i1>) -> tensor<i1>
    // CHECK: "tf.TPUReplicatedInput"
    %4, %5 = tf_executor.island(%1) wraps "tf.TPUReplicatedInput"(%2) {index = -1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %0 : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @preserve_unreachable_tpu_replicated_output
func.func @preserve_unreachable_tpu_replicated_output(%arg0: tensor<i1>) {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    // CHECK: "tf.NoOp"
    %1 = tf_executor.island wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"
    %2, %3 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i1>) -> tensor<i1>
    // CHECK: "tf.TPUReplicatedOutput"
    %4, %5 = tf_executor.island(%1) wraps "tf.TPUReplicatedOutput"(%2) : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %0 : !tf_executor.control
  }
  func.return
}

// CHECK-LABEL: func @preserve_unreachable_custom_op
func.func @preserve_unreachable_custom_op(%arg0: tensor<i1>) {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    // CHECK: "tf.NoOp"
    %1 = tf_executor.island wraps "tf.NoOp"() : () -> ()
    // CHECK: "tf.Identity"
    %2, %3 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i1>) -> tensor<i1>
    // CHECK: "tf.CustomOp"
    %4, %5 = tf_executor.island(%1) wraps "tf.CustomOp"(%2) : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %0 : !tf_executor.control
  }
  func.return
}
