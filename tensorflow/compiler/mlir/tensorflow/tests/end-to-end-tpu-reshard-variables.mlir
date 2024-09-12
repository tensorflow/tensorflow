// RUN: tf-opt %s -tf-replicated-clustering-bridge-v2 -tfrt-lower-cluster-to-runtime-ops-tpu 2>&1 | FileCheck %s

// TPUReshardVariables should be inserted even when While functions' shapes are
// different than While operand shapes. Test the whole tf-tpu-bridge because
// correct insertion of TPUReshardVariables depends on multiple passes including
// TPUVariableRuntimeReformatting, ShapeInference, WhileRegion canonicalization,
// and TPUMergeVariablesWithExecute.

// CHECK-LABEL: module
// CHECK: tf_device.replicate
// CHECK:   TPUReshardVariables
// CHECK:   TPUExecuteAndUpdateVariables
// CHECK: tf_device.replicate
// CHECK:   TPUReshardVariables
module attributes {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:CPU:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU:1" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1163 : i32}} {
  func.func @main(%arg0: tensor<*x!tf_type.resource> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) {
    tf_executor.graph {
      %control = tf_executor.island {
        "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @partitioned}> : (tensor<*x!tf_type.resource>) -> ()
        tf_executor.yield
      }
      tf_executor.fetch %control : !tf_executor.control
    }
    return
  }
  func.func private @partitioned(%arg0: tensor<*x!tf_type.resource> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) {
    tf_executor.graph {
      %control = tf_executor.island {
        %0 = "tf.While"(%arg0) <{body = @while_body, cond = @while_cond, is_stateless = false, shape_invariant}> : (tensor<*x!tf_type.resource>) -> tensor<!tf_type.resource<tensor<i32>>>
        tf_executor.yield
      }
      tf_executor.fetch %control : !tf_executor.control
    }
    return
  }
  func.func private @while_body(%arg0: tensor<!tf_type.resource<tensor<i32>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> tensor<!tf_type.resource<tensor<i32>>> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        "tf.TPUReplicateMetadata"() <{allow_soft_placement = true, computation_shape = [], device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 2 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true}> {_tpu_replicate = "cluster", device = ""} : () -> ()
        %cst = "tf.Const"() <{value = dense<1> : tensor<i32>}> {_tpu_replicate = "cluster"} : () -> tensor<*xi32>
        %1 = "tf.TPUReplicatedInput"(%arg0) <{index = -1 : i64, is_mirrored_variable = true, is_packed = true}> : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
        %2 = "tf.ReadVariableOp"(%1) {_tpu_replicate = "cluster", device = ""} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<*xi32>
        %3 = "tf.Add"(%2, %cst) {_tpu_replicate = "cluster"} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
        "tf.AssignVariableOp"(%1, %3) {_tpu_replicate = "cluster"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<*xi32>) -> ()
        %4:2 = "tf.TPUReplicatedOutput"(%1) : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>)
        tf_executor.yield %arg0 : tensor<!tf_type.resource<tensor<i32>>>
      }
      tf_executor.fetch %outputs : tensor<!tf_type.resource<tensor<i32>>>
    }
    return %0 : tensor<!tf_type.resource<tensor<i32>>>
  }
  func.func private @while_cond(%arg0: tensor<!tf_type.resource<tensor<i32>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> tensor<*xi1> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        %cst = "tf.Const"() <{value = dense<false> : tensor<i1>}> : () -> tensor<*xi1>
        tf_executor.yield %cst : tensor<*xi1>
      }
      tf_executor.fetch %outputs : tensor<*xi1>
    }
    return %0 : tensor<*xi1>
  }
}
