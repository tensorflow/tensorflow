// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-embedding-pipelining | FILECHECK_OPTS="" FileCheck %s

// This test verifies the handling of TPU replicated inputs and outputs as well as the extraction of the four main functions.
module {
  func.func @main() {
    %cst_main = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst_main) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // Verify that everything is extracted into one of the four functions.
    // The order of these functions is also significant.
    // CHECK: {{.*StatefulPartitionedCall.* f = @_func_non_tpu.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_sc_forward.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_core_tpu.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_sc_backward.*}}
    // CHECK-NEXT: return
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
  // Generated functions
  // non_tpu should have to TPU ops - just identity and return (in this test).
  // CHECK: func.func private @_func_non_tpu
  // CHECK-NEXT: tf.Identity
  // CHECK-NEXT: return

  // sc_forward should have TPU ops including replicated outputs but not inputs
  // CHECK: func.func private @_func_sc_forward
  // CHECK-NOT: TPUReplicatedInput
  // CHECK-DAG: TPUReplicateMetadata
  // CHECK-DAG: TPUCompilationResult
  // CHECK-DAG: TPUReplicatedOutput
  // CHECK: return

  // core_tput should have TPU ops including both replicated inputs and outputs
  // CHECK: func.func private @_func_core_tpu
  // CHECK-DAG: TPUReplicatedInput
  // CHECK-DAG: TPUReplicateMetadata
  // CHECK-DAG: TPUCompilationResult
  // CHECK-DAG: TPUReplicatedOutput
  // CHECK: return

  // sc_backward should have TPU ops including replicted inputs but not outputs
  // CHECK: func.func private @_func_sc_backward
  // CHECK-NOT: TPUReplicatedOutput
  // CHECK-DAG: TPUReplicateMetadata
  // CHECK-DAG: TPUCompilationResult
  // CHECK-DAG: TPUReplicatedInput
  // CHECK: return
}

// -----
// This test verifies that the extraction works correctly for evaluation-only models.
module {
  func.func @main() {
    %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.While"(%cst) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>) -> (tensor<i32>) {
    // CHECK: {{.*StatefulPartitionedCall.* f = @_func_non_tpu.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_sc_forward.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_core_tpu.*}}
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
  // Only verify sc_backward. The previous test case verifies everything else.
  // CHECK: func.func private @_func_sc_backward
  // CHECK-NEXT: return
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
// A test verifying TPUReplicatedOutput in the input graph doesn't trigger
// any additional TPUReplicatedInput or TPUReplicatedOutput ops.
module {
  func.func @main() {
    %cst_1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_2 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %0:2 = "tf.While"(%cst_1, %cst_2) {body = @while_body, cond = @while_cond, is_stateless = false} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    return
  }
  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    // CHECK: {{.*StatefulPartitionedCall.* f = @_func_non_tpu.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_sc_forward.*}}
    // CHECK-NEXT: {{.*StatefulPartitionedCall.* f = @_func_core_tpu.*}}
    // metadata ops
    "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _replication_info = "repl_info", num_replicas = 2 : i64} : () -> ()
    %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "repl_info"} : () -> tensor<!tf_type.string>
    %2 = "tf.Const"() {_embedding_pipelining = "forward", _replication_info = "repl_info", value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3:2 = "tf.TPUReplicatedOutput"(%2) {device = ""} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)

    // core_tpu ops:
    %res_t = "tf.Const"() {_replication_info = "repl_info", value = dense<4> : tensor<i32>} : () -> tensor<i32>

    // non_tpu_ops
    %res_n = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>

    return %res_n, %3#1 : tensor<i32>, tensor<i32>
  }
  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Less"(%arg1, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  // CHECK-DAG: TPUReplicatedOutput
  // CHECK-NOT: TPUReplicatedoutput
  // CHECK-NOT: TPUReplicatedInput
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
