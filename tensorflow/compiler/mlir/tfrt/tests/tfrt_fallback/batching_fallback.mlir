// RUN: tfrt_fallback_translate --mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd:1,1 | FileCheck %s
// RUN: tfrt_fallback_translate --mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd:8 | FileCheck %s

func.func @matmul_cpu(%ch: !tfrt.chain, %a: !tfrt_fallback.tf_tensor, %b: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  // Enqueue a sleep onto blocking work queue, %ch0 is fulfilled when sleeping is done.
  %us = tfrt.constant.i32 1000
  %ch0 = "tfrt_test.blocking.usleep"(%us) : (i32) -> (!tfrt.chain)

  %ch1 = tfrt.merge.chains %ch, %ch0 : !tfrt.chain, !tfrt.chain

  %ch2 = tfrt_fallback_async.createop(%ch1) key(0) device("/CPU:0") "tf.MatMul"() {T = i32} num_args(2)

  %ch3, %result = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("/CPU:0") "tf.MatMul"(%a, %b) {T = i32}  : 1

  %s = "tfrt_test.get_string"() { value = "Running @matmul_cpu" } : () -> !tfrt.string
  %ch4 = "tfrt_test.print_string"(%s, %ch3) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  %ch5 = "tfrt_fallback_async.print_tensor"(%result, %ch4) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch5, %result : !tfrt.chain, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: --- Running 'batch_function_fallback_concat_test'
func.func @batch_function_fallback_concat_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a1 = tfrt_fallback_async.const_dense_tensor dense<[[2, 2], [2, 2]]> : tensor<2x2xi32>
  %a2 = tfrt_fallback_async.const_dense_tensor dense<[[3, 3], [3, 3]]> : tensor<2x2xi32>
  %b = tfrt_fallback_async.const_dense_tensor dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>

  // Two batch_size=2 batches get concatenated.
  %result_1 = tfrt_fallback_async.batch_function device("/device:CPU:0") @matmul_cpu (%a1, %b) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 1000000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [i32],
      Tout = [i32]} : 1

  %result_2 = tfrt_fallback_async.batch_function device("/device:CPU:0") @matmul_cpu (%a2, %b) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 1000000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [i32],
      Tout = [i32]} : 1

  // Since batch function kernel scheduling is async, the above 2 batches can arrive in any order.
  // CHECK: Running @matmul_cpu
  // CHECK-NEXT: Tensor<type: int32 shape: [4,2] values: [[value_output:.*]]>

  // CHECK: Tensor<type: int32 shape: [2,2] values: [4 4]
  %ch1 = "tfrt_fallback_async.print_tensor"(%result_1, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: Tensor<type: int32 shape: [2,2] values: [6 6]
  %ch2 = "tfrt_fallback_async.print_tensor"(%result_2, %ch1) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'batch_function_fallback_timeout_test'
func.func @batch_function_fallback_timeout_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_dense_tensor dense<[[4, 4], [4, 4]]> : tensor<2x2xi32>
  %b = tfrt_fallback_async.const_dense_tensor dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>

  // One batch_size=2 batches get padded and processed after timeout.
  %result = tfrt_fallback_async.batch_function device("/device:CPU:0") @matmul_cpu (%a, %b) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 1000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [i32],
      Tout = [i32]} : 1

  // CHECK: Running @matmul_cpu
  // CHECK-NEXT: Tensor<type: int32 shape: [4,2] values: [[value_output:.*]]>
  // CHECK: Tensor<type: int32 shape: [2,2] values: [8 8]
  %ch1 = "tfrt_fallback_async.print_tensor"(%result, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'batch_function_fallback_no_padding_test'
func.func @batch_function_fallback_no_padding_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_dense_tensor dense<[[4, 4], [4, 4]]> : tensor<2x2xi32>
  %b = tfrt_fallback_async.const_dense_tensor dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>

  // One batch_size=2 batches get processed after timeout.
  %result = tfrt_fallback_async.batch_function device("/device:CPU:0") @matmul_cpu (%a, %b) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 1000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      disable_padding = true,
      Tin = [i32],
      Tcaptured = [i32],
      Tout = [i32]} : 1

  // CHECK: Running @matmul_cpu
  // As no padding is appended, the tensor shape printed inside the batch function
  // is [2,2] instead of [4,2]
  // CHECK-NEXT: Tensor<type: int32 shape: [2,2] values: [[value_output:.*]]>

  // CHECK: Tensor<type: int32 shape: [2,2] values: [8 8]
  %ch1 = "tfrt_fallback_async.print_tensor"(%result, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// This test is for testing support for Adaptive Batch Scheduler, and it is
// triggered by num_batch_threads<=0
// CHECK-LABEL: --- Running 'batch_function_fallback_zero_batch_thread_test'
func.func @batch_function_fallback_zero_batch_thread_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_dense_tensor dense<[[2, 2], [2, 2]]> : tensor<2x2xi32>
  %b = tfrt_fallback_async.const_dense_tensor dense<[[2, 2], [2, 2]]> : tensor<2x2xi32>

  // One batch_size=2 batches get padded and processed after timeout.
  %result = tfrt_fallback_async.batch_function device("/device:CPU:0") @matmul_cpu (%a, %b) {
      num_batch_threads = 0,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 1000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [i32],
      Tout = [i32]} : 1

  // CHECK: Running @matmul_cpu
  // CHECK-NEXT: Tensor<type: int32 shape: [4,2] values: [[value_output:.*]]>
  // CHECK: Tensor<type: int32 shape: [2,2] values: [8 8]
  %ch1 = "tfrt_fallback_async.print_tensor"(%result, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// Test to verify the fix for b/190394141:
// Given a function that returns multiple values referenced to the same value,
func.func @returns_multiple_refs(%ch: !tfrt.chain, %a: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) {
  tfrt.return %ch, %a, %a : !tfrt.chain, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// ... when the function returns_multiple_refs is invoked by the batch_function,
// ... then the code should not crash.
// CHECK-LABEL: Running 'test_batch_returns_multiple_refs'
func.func @test_batch_returns_multiple_refs() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %0 = tfrt_fallback_async.const_dense_tensor dense<[[1, 1], [1, 1]]> : tensor<2x2xi32>

  %1, %2 = tfrt_fallback_async.batch_function device("/device:CPU:0") @returns_multiple_refs (%0) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 10000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [],
      Tout = [i32, i32]} : 2

  %3, %4 = tfrt_fallback_async.batch_function device("/device:CPU:0") @returns_multiple_refs (%0) {
      num_batch_threads = 1,
      max_batch_size = 4,
      allowed_batch_sizes = [4],
      batch_timeout_micros = 10000,
      container = "container",
      shared_name = "shared_name",
      batching_queue = "batching_queue",
      enable_large_batch_splitting = false,
      Tin = [i32],
      Tcaptured = [],
      Tout = [i32, i32]} : 2


  %ch1 = "tfrt_fallback_async.print_tensor"(%1, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch2 = "tfrt_fallback_async.print_tensor"(%3, %ch1) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

