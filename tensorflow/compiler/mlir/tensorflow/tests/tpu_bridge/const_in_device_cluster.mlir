// RUN: tf-opt %s -tf-tpu-bridge | FileCheck %s

// Verifies that constants defined within a cluster is extracted out of the
// cluster. They shouldn't placed on the device if they are not used on the
// device and are directly returned.

// Alternatively, a python test can be used for verification of pass pipeline
// to avoid manual modifications of this test. Consider removing this test if
// this has some maintenance overhead to match input invariants provided by the
// front-end.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:1" = {}, "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 725 : i32}}  {

func @main(%arg11: tensor<*x!tf.resource<tensor<4xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg12: tensor<*x!tf.resource<tensor<4xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<*xf32>, tensor<*x!tf.string>) attributes {tf.entry_function = {control_outputs = "", inputs = "arg0,arg1"}} {
  %0:2 = tf_executor.graph {
    %ordinal, %control = tf_executor.island {
      %1 = "tf.TPUOrdinalSelector"() {device = ""} : () -> tensor<?xi32>
      tf_executor.yield %1 : tensor<?xi32>
    }
    %outputs:2, %control_25 = tf_executor.island {
      %1:2 = "tf.TPUPartitionedCall"(%arg11, %arg12, %ordinal) {autotuner_thresh = 0 : i64, device = "", f = @__subgraph} : (tensor<*x!tf.resource<tensor<4xf32>>>, tensor<*x!tf.resource<tensor<4xf32>>>, tensor<?xi32>) -> (tensor<*xf32>, tensor<*x!tf.string>)
      tf_executor.yield %1#0, %1#1 : tensor<*xf32>, tensor<*x!tf.string>
    }
    tf_executor.fetch %outputs#0, %outputs#1 : tensor<*xf32>, tensor<*x!tf.string>
  }
  return %0#0, %0#1 : tensor<*xf32>, tensor<*x!tf.string>
}

// Simple graph to add the given two resources. This makes sure that the cluster is not empty and not elided.

// CHECK-LABEL: @__subgraph
func private @__subgraph(%arg0: tensor<*x!tf.resource>, %arg1: tensor<*x!tf.resource>) -> (tensor<*xf32>, tensor<*x!tf.string>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
  %0:2 = tf_executor.graph {
    %control = tf_executor.island {
      "tf.NoOp"() {_pivot_for_cluster = "cluster", device = ""} : () -> ()
      tf_executor.yield
    }
    %control_1 = tf_executor.island(%control) {
      "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
      tf_executor.yield
    }
    // CHECK: "tf.Const"() {value = dense<"string_const_value">
    // CHECK: tf_device.launch
    // CHECK: "tf._TPUCompileMlir"
    %outputs, %control_2 = tf_executor.island(%control_1) {
      %1 = "tf.Const"() {_tpu_replicate = "cluster", _xla_outside_compilation = "0", device = "", value = dense<"string_const_value"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      tf_executor.yield %1 : tensor<!tf.string>
    }
    %outputs_3, %control_4 = tf_executor.island {
      %1 = "tf.Identity"(%outputs) {_tpu_output_identity = true, _tpu_replicate = "cluster", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_5, %control_6 = tf_executor.island {
      %1 = "tf.TPUReplicatedOutput"(%outputs_3) {device = ""} : (tensor<*x!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_7, %control_8 = tf_executor.island(%control) {
      %1 = "tf.Identity"(%outputs_5) {device = ""} : (tensor<*x!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_9, %control_10 = tf_executor.island {
      %1 = "tf.Identity"(%outputs) {_tpu_output_identity = true, _tpu_replicate = "cluster", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_11, %control_12 = tf_executor.island {
      %1 = "tf.TPUReplicatedOutput"(%outputs_9) {device = ""} : (tensor<*x!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_13, %control_14 = tf_executor.island(%control) {
      %1 = "tf.Identity"(%outputs_11) {device = ""} : (tensor<*x!tf.string>) -> tensor<*x!tf.string>
      tf_executor.yield %1 : tensor<*x!tf.string>
    }
    %outputs_27, %control_28 = tf_executor.island(%control_1) {
      %1 = "tf.ReadVariableOp"(%arg0) {_tpu_replicate = "cluster", device = ""} : (tensor<*x!tf.resource>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    %outputs_28, %control_29 = tf_executor.island(%control_1) {
      %1 = "tf.ReadVariableOp"(%arg1) {_tpu_replicate = "cluster", device = ""} : (tensor<*x!tf.resource>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    %outputs_29, %control_30 = tf_executor.island {
      %1 = "tf.Add"(%outputs_27, %outputs_28) {_tpu_replicate = "cluster", device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    %outputs_51, %control_52 = tf_executor.island {
      %1 = "tf.Identity"(%outputs_29) {_tpu_output_identity = true, _tpu_replicate = "cluster", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    %outputs_53, %control_54 = tf_executor.island {
      %1 = "tf.TPUReplicatedOutput"(%outputs_51) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    %outputs_55, %control_56 = tf_executor.island(%control) {
      %1 = "tf.Identity"(%outputs_53) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %1 : tensor<*xf32>
    }
    tf_executor.fetch %outputs_55, %outputs_7 : tensor<*xf32>, tensor<*x!tf.string>
  }
  return %0#0, %0#1 : tensor<*xf32>, tensor<*x!tf.string>
}

}
