// RUN: tfrt_fallback_translate -mlir-to-bef %s                                \
// RUN:   | tf_bef_executor --test_init_function=register_op_handlers_kernel   \
// RUN:                     --host_allocator_type=malloc                       \
// RUN:                     --work_queue_type=mstd:72                          \
// RUN:   | FileCheck %s --dump-input=always

// A set of benchmarks for large tensor inputs. These benchmarks measure
// codegen/runtime efficiency for executing parallel/concurrent code. Fallback
// kernels rely on Eigen for parallelizing compute operation.

module @kernels attributes { tfrt.compiled } {
  func.func @log1p(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = "tf.Log1p"(%arg0): (tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %0 : tensor<?x?xf32>
  }
}

// CHECK: --- Running 'BM_fallback_log1p_f32'
func.func @BM_fallback_log1p_f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(1) device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<1.0> : tensor<1024x1024xf32>
     } num_args(0)

  %ch2 = tfrt_fallback_async.createop(%ch1) key(2) device("/CPU:0")
    "tf.Log1p"() { T = f32 } num_args(1)

  %ch3, %const = tfrt_fallback_async.executeop.seq(%ch2) key(1) cost(100)
                                                         device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<1.0> : tensor<1024x1024xf32>
     } : 1

  %ch4 = tfrt_test.benchmark "BM_fallback_log1p_f32"(
    %const    : !tfrt_fallback.tf_tensor,
    %ch3      : !tfrt.chain
  )
  duration_secs = 1, max_count = 10000, num_warmup_runs = 10
  {
    %result = tfrt_fallback_async.executeop key(2) cost(100) device("/CPU:0")
       "tf.Log1p"(%const) { T = f32 } : 1
    tfrt.return %result : !tfrt_fallback.tf_tensor
  }

  tfrt.return %ch4 : !tfrt.chain
}
