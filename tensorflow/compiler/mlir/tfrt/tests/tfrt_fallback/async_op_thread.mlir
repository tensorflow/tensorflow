// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd 2>&1 | FileCheck %s

func.func @test_async_op_kernel_thread() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("/CPU:0") "tf.Const"()
         { dtype = i32, value = dense<[2]> : tensor<1xi32> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("/CPU:0")
         "tf.TestAsyncTfrtAsyncThread"() {T = i32} num_args(1)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(2) device("/CPU:0")
         "tf.TestPrintThreadName"() num_args(0)

  %ch4, %0 = tfrt_fallback_async.executeop.seq(%ch3) key(0) cost(100)
             device("/CPU:0") "tf.Const"()
             { dtype = i32, value = dense<[2]> : tensor<1xi32> } : 1
  // Given TestAsyncTfrtAsyncThread will invoke done callback in thread with
  // name "test_thread_in_compute_async",
  // CHECK: TestAsyncTfrtAsyncThread thread name: test_thread_in_compute_async
  %ch5, %1 = tfrt_fallback_async.executeop.seq(%ch4) key(1) cost(100)
             device("/CPU:0") "tf.TestAsyncTfrtAsyncThread"(%0) {T = i32} : 1
  // ... when TestPrintThreadName is part of the next op in sequence of
  //     TestAsyncTfrtAsyncThread op,
  // ... then TestPrintThreadName should not run in the thread in
  //     TestAsyncTfrtAsyncThread.
  // CHECK-NOT: TestPrintThreadName thread name: test_thread_in_compute_async
  %ch6 = tfrt_fallback_async.executeop.seq(%ch5) key(2) cost(100)
             device("/CPU:0") "tf.TestPrintThreadName"() : 0

  tfrt.return %ch6: !tfrt.chain
}
