// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-embedding-pipelining | FILECHECK_OPTS="" FileCheck %s

// This test verifies the handling of TPU replicated inputs and outputs as well as the extraction of the four main functions.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // Verify the overall pipelining control flow and supporting functions.
    // The order of these functions is also significant.
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @non_tpu.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @start_step_0.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @non_tpu.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @start_step_1.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*tf.While.* <{body = @new_while_body.* cond = @new_while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @finish_step_nm2.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @finish_step_nm1.*}}
    // CHECK: return
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  // Generated functions for control flow ops (if, while, switch)

  // non_tpu should have to TPU ops - just identity and return (in this test).
  // CHECK: func.func private @non_tpu
  // CHECK-NEXT: tf.Identity
  // CHECK-NEXT: return

  // Since there is a backward pass, finish_step_nm2 should be non-empty.
  // CHECK: func.func private @finish_step_nm2
  // CHECK-NEXT: tf.TPUReplicateMetadata
}

// -----
// This test verifies that the pipelining works correctly for evaluation-only models.
module {
  func.func @main() {
    %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // The pipelining control flow and supporting functions stay the same as the training version above.
    // The order of these functions is also significant.
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @non_tpu.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @start_step_0.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @non_tpu.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @start_step_1.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @while_cond.*}}
    // CHECK: {{.*tf.While.* <{body = @new_while_body.* cond = @new_while_cond.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @finish_step_nm2.*}}
    // CHECK: {{.*StatefulPartitionedCall.* f = @finish_step_nm1.*}}
    // CHECK: return
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 1 : i64} : () -> ()
    %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Identity"(%arg0) {_embedding_pipelining = "forward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  // There's no backward pass so finish_step_nm2 should be empty
  // CHECK: func.func private @finish_step_nm2
  // CHECK-NEXT: return
}

// -----
// This test verifies that the new WhileOp inherrits the parallel_iterations attribute.
module {
  func.func @main() {
    %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst) {body = @while_body, cond = @while_cond, is_stateless = false, parallel_iterations = 3} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // The pipelining control flow and supporting functions stay the same as the training version above.
    // The order of these functions is also significant.
    // CHECK: {{.*tf.While.* <{body = @new_while_body.* cond = @new_while_cond.* parallel_iterations = 3}}
    // CHECK: return
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 1 : i64} : () -> ()
    %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Identity"(%arg0) {_embedding_pipelining = "forward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// A test verifying too many TPUReplicateMetadataOp ops. Same logic tests too many TPUCompilationResultOp ops.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource<tensor<512x256xf32>>>) {
    %cst = "tf.Const"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %0 = "tf.While"(%cst) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i1>) -> (tensor<i1>)
    return
  }
  // expected-error @+1 {{number of tf.TPUReplicateMetadata in loop body is not 1}}
  func.func private @while_body(%arg0: tensor<i1>) -> (tensor<i1>) {
    // metadata ops
    %embedding_pass_trigger = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 1 : i64} : () -> ()
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 1 : i64} : () -> ()
    %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>
    return %arg0 : tensor<i1>
  }
  func.func private @while_cond(%arg0: tensor<i1>) -> tensor<i1> {
    return %arg0 : tensor<i1>
  }
}

// -----
// A test verifying the replication region of TPUReplicateMetadataOp ops. Same logic tests too many TPUCompilationResultOp ops.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource<tensor<512x256xf32>>>) {
    %cst = "tf.Const"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %0 = "tf.While"(%cst) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i1>) -> (tensor<i1>)
    return
  }
  func.func private @while_body(%arg0: tensor<i1>) -> (tensor<i1>) {
    // metadata ops
    %embedding_pass_trigger = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 1 : i64} : () -> ()
    // expected-error @+1 {{'tf.TPUCompilationResult' op is not part of the replication region "repl_info" vs "wrong_repl_info"}}
    %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "wrong_repl_info"} : () -> tensor<!tf_type.string>
    return %arg0 : tensor<i1>
  }
  func.func private @while_cond(%arg0: tensor<i1>) -> tensor<i1> {
    return %arg0 : tensor<i1>
  }
}

// -----
// Verify error for backward pass with no forward pass.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    // expected-error @+1 {{'tf.Identity' op embedding backwards pass op with no forwards pass ops}}
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// Verify error for unknown _embedding_pipelining attribute value.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    // expected-error @+1 {{'tf.Identity' op embedding op has unknown _embedding_pipelining attribute value garbage.}}
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "garbage", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// Verify error for multiple WhileOp use of while_body function.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    // expected-error @+1 {{'tf.While' op multiple users of function.}}
    %1 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// Verify error for non-WhileOp use of while_body function.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    // expected-error @+1 {{'tf.StatefulPartitionedCall' op non while use of function.}}
    %38 = "tf.StatefulPartitionedCall"(%cst_main) {config = "", config_proto = "", executor_type = "", f = @while_body} : (tensor<i32>) -> tensor<i32>
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// Verify one while body function per while loop op.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    // expected-error @+1 {{'tf.While' op multiple users of function.}}
    %1 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// Verify that the function to be pipelined is a while loop body function.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    return
  }
  // expected-error @+1 {{'func.func' op unable to find while body user.}}
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// This test verifies that TPUReplicatedInputOps for resource variable args are packed.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0, %1 = "tf.While"(%cst_main, %arg0) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>)
    return
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) {
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // expected-error @+1 {{'tf.TPUReplicatedInput' op unexpected variable input, not packed}}
    %37 = "tf.TPUReplicatedInput"(%arg1) {device = "", index = -1 : i64, is_mirrored_variable = true, is_packed = false} : (tensor<*x!tf_type.resource<tensor<i64>>>) -> tensor<*x!tf_type.resource<tensor<i64>>>

   // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    %cst_12 = "tf.Const"() {_replication_info = "repl_info", _xla_compile_device_type = "TPU", device = "", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    "tf.AssignAddVariableOp"(%37, %cst_12) {_has_manual_control_dependencies = true, _replication_info = "while/cluster_while_body_451", _xla_compile_device_type = "TPU", device = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()

    return %res_n, %arg1 : tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>
  }
  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// This test verifies that duplicate TPUReplicatedInput ops for a resource variable arg is an error.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0, %1 = "tf.While"(%cst_main, %arg0) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>)
    return
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) {
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // expected-error @+1 {{'tf.TPUReplicatedInput' op unexpected multiple TPUReplicatedInputOp for single argument}}
    %37 = "tf.TPUReplicatedInput"(%arg1) {device = "", index = -1 : i64, is_mirrored_variable = true, is_packed = true} : (tensor<*x!tf_type.resource<tensor<i64>>>) -> tensor<*x!tf_type.resource<tensor<i64>>>
    %38 = "tf.TPUReplicatedInput"(%arg1) {device = "", index = -1 : i64, is_mirrored_variable = true, is_packed = true} : (tensor<*x!tf_type.resource<tensor<i64>>>) -> tensor<*x!tf_type.resource<tensor<i64>>>

   // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    %cst_12 = "tf.Const"() {_replication_info = "repl_info", _xla_compile_device_type = "TPU", device = "", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    "tf.AssignAddVariableOp"(%37, %cst_12) {_has_manual_control_dependencies = true, _replication_info = "while/cluster_while_body_451", _xla_compile_device_type = "TPU", device = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()

    return %res_n, %arg1 : tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>
  }
  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// This test verifies the EliminateResourceLoops workaround.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0, %1 = "tf.While"(%cst_main, %arg0) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>)
    return
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>) {
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    %rsrc_copy = "tf.StatefulPartitionedCall"(%arg1) {f = @broken_func, config = "", config_proto = "", executor_type = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>)  -> (tensor<*x!tf_type.resource<tensor<i64>>>)
    // We expect uses of %rsrc_copy are replaced by the input resource variable (%arg1 in this context).
    "tf.StatefulPartitionedCall"(%arg1) {f = @func1, config = "", config_proto = "", executor_type = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>)  -> ()
    "tf.StatefulPartitionedCall"(%rsrc_copy) {f = @func2, config = "", config_proto = "", executor_type = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>)  -> ()

   // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    %cst_12 = "tf.Const"() {_replication_info = "repl_info", _xla_compile_device_type = "TPU", device = "", value = dense<1> : tensor<i64>} : () -> tensor<i64>

    return %res_n, %arg1 : tensor<i32>, tensor<*x!tf_type.resource<tensor<i64>>>
  }
  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"}) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @broken_func(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"})  -> (tensor<*x!tf_type.resource<tensor<i64>>>) {
    %x = "tf.Identity"(%arg0) : (tensor<*x!tf_type.resource<tensor<i64>>>)  -> (tensor<*x!tf_type.resource<tensor<i64>>>)
    %y = "tf.Identity"(%x) : (tensor<*x!tf_type.resource<tensor<i64>>>)  -> (tensor<*x!tf_type.resource<tensor<i64>>>)
    return %y : tensor<*x!tf_type.resource<tensor<i64>>>
  }
  func.func private @func1(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"})  -> () {
    return
  }
  func.func private @func2(%arg0: tensor<*x!tf_type.resource<tensor<i64>>> {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf._user_specified_name = "rsrc", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0"})  -> () {
    return
  }
  // Make sure func1 and func2 use the original resource variable and not the result of @broken_func.
  // CHECK: func.func private @non_tpu
  // CHECK: {{.*%0 = \"tf.StatefulPartitionedCall\"\(%arg0\).*f = @broken_func.*}}
  // CHECK: {{.*StatefulPartitionedCall\"\(%arg0\).*f = @func1.*}}
  // CHECK: {{.*StatefulPartitionedCall\"\(%arg0\).*f = @func2.*}}
}

// -----
// This test verifies that the pipelining pass has no effect when tf.WriteSummaryOp is present.
module {
  func.func @main(%arg0: tensor<*x!tf_type.resource>) {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0:2 = "tf.While"(%cst_main, %arg0) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>, tensor<*x!tf_type.resource>) -> (tensor<i32>, tensor<*x!tf_type.resource>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource>) -> (tensor<i32>, tensor<*x!tf_type.resource>) {
    // CHECK-NOT: {{.*tf.While.* body = @new_while_body.* cond = @new_while_cond.*}}
    // CHECK: return
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops:
    %res_t = "tf.Identity"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // backward_ops
    %res_b = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>

    %tensor_int64 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %tensor_string = "tf.Const"() {value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    "tf.WriteSummary"(%arg1, %tensor_int64, %tensor_int64, %tensor_string, %tensor_string): (tensor<*x!tf_type.resource>, tensor<i64>, tensor<i64>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()

    return %res_n, %arg1 : tensor<i32>, tensor<*x!tf_type.resource>
  }
  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// -----
// This test verifies that for ops with multiple TPU -> backward edges, we
// create input/output ops for all of them.

// CHECK-LABEL: @multiple
module @multiple {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  // CHECK-LABEL: while_body
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %comp_res = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>

    // forward_ops
    %res_f = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<2> : tensor<i32>} : () -> tensor<i32>

    // core_tpu ops
    %res_t = "tf.Cast"(%res_f) {_replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>
    // CHECK: [[input:%.*]] = "tf.TPUReplicatedInput"
    // CHECK: "tf.Identity"([[input]])
    // CHECK: "tf.Identity"([[input]])
    // CHECK: "tf.Identity"([[input]])
    // CHECK: "tf.Identity"([[input]])
    // backward_ops
    %res_b1 = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>
    %res_b2 = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>
    %res_b3 = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>
    %res_b4 = "tf.Identity"(%res_t) {_embedding_pipelining = "backward", _replication_info = "repl_info"} : (tensor<i32>) -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Add"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

    return %res_n : tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
