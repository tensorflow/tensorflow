// RUN: tf-opt %s -tf-tpu-bridge 2>&1 | FileCheck %s

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
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%res : tensor<*x!tf_type.resource>):
    "tf_executor.graph"() ({
      %ctrl = "tf_executor.island"() ({
        "tf.StatefulPartitionedCall"(%res) {config = "", config_proto = "", executor_type = "", f = @partitioned} : (tensor<*x!tf_type.resource>) -> ()
        "tf_executor.yield"() : () -> ()
      }) : () -> (!tf_executor.control)
      "tf_executor.fetch"(%ctrl) : (!tf_executor.control) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {arg_attrs = [{tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}], function_type = (tensor<*x!tf_type.resource>) -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  ^bb0(%res : tensor<*x!tf_type.resource>):
    "tf_executor.graph"() ({
      %ctrl = "tf_executor.island"() ({
        %w = "tf.While"(%res) {body = @while_body, cond = @while_cond, is_stateless = false, shape_invariant} : (tensor<*x!tf_type.resource>) -> (tensor<!tf_type.resource<tensor<i32>>>)
        "tf_executor.yield"() : () -> ()
      }) : () -> (!tf_executor.control)
      "tf_executor.fetch"(%ctrl) : (!tf_executor.control) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {arg_attrs = [{tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}], function_type = (tensor<*x!tf_type.resource>) -> (), sym_name = "partitioned", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%res : tensor<!tf_type.resource<tensor<i32>>>):
    %g = "tf_executor.graph"() ({
      %i:2 = "tf_executor.island"() ({
        "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", allow_soft_placement = true, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 2 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
        %one = "tf.Const"() {_tpu_replicate = "cluster", value = dense<1> : tensor<i32>} : () -> tensor<*xi32>
        %res_rep = "tf.TPUReplicatedInput"(%res) {index = -1 : i64, is_mirrored_variable = true, is_packed = true} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
        %read = "tf.ReadVariableOp"(%res_rep) {_tpu_replicate = "cluster", device = ""} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<*xi32>
        %inc = "tf.Add"(%read, %one) {_tpu_replicate = "cluster"} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
        "tf.AssignVariableOp"(%res_rep, %inc) {_tpu_replicate = "cluster"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<*xi32>) -> ()
        %res_out:2 = "tf.TPUReplicatedOutput"(%res_rep) : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>)
        "tf_executor.yield"(%res) : (tensor<!tf_type.resource<tensor<i32>>>) -> ()
      }) : () -> (tensor<!tf_type.resource<tensor<i32>>>, !tf_executor.control)
      "tf_executor.fetch"(%i#0) : (tensor<!tf_type.resource<tensor<i32>>>) -> ()
    }) : () -> tensor<!tf_type.resource<tensor<i32>>>
    "func.return"(%g) : (tensor<!tf_type.resource<tensor<i32>>>) -> ()
  }) {arg_attrs = [{tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}], function_type = (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>, sym_name = "while_body", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%res : tensor<!tf_type.resource<tensor<i32>>>):
    %g = "tf_executor.graph"() ({
      %i:2 = "tf_executor.island"() ({
        %c = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<*xi1>
        "tf_executor.yield"(%c) : (tensor<*xi1>) -> ()
      }) : () -> (tensor<*xi1>, !tf_executor.control)
      "tf_executor.fetch"(%i#0) : (tensor<*xi1>) -> ()
    }) : () -> tensor<*xi1>
    "func.return"(%g) : (tensor<*xi1>) -> ()
  }) {arg_attrs = [{tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}], function_type = (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<*xi1>), sym_name = "while_cond", sym_visibility = "private"} : () -> ()
}) {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:CPU:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU:0" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU:1" = {}, "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1163 : i32}} : () -> ()
