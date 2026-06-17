// RUN: tfrt_fallback_translate -mlir-to-bef %s                                \
// RUN:   | tf_bef_executor --test_init_function=register_op_handlers_kernel   \
// RUN:   | FileCheck %s --dump-input=always

// A set of benchmarks measuring runtime/compiler overheads for running
// kernels with small inputs. These benchmarks do not launch any concurrent
// tasks and do not use concurrent work queue or intra-op thread pool.

module @kernels attributes { tfrt.compiled } {
  func.func @rsqrt(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    func.return %0 : tensor<?xf32>
  }

  func.func @rsqrt_tanh(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    %1 = "tf.Tanh"(%0): (tensor<?xf32>) -> tensor<?xf32>
    func.return %1 : tensor<?xf32>
  }
}

// CHECK: --- Running 'BM_fallback_rsqrt_f32'
func.func @BM_fallback_rsqrt_f32() {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(1) device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
     } num_args(0)

  %ch2 = tfrt_fallback_async.createop(%ch1) key(2) device("/CPU:0")
    "tf.Rsqrt"() { T = f32 } num_args(1)

  %ch3, %const = tfrt_fallback_async.executeop.seq(%ch1) key(1) cost(100)
                                                         device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
     } : 1

  tfrt_test.benchmark "BM_fallback_rsqrt_f32"(
    %const    : !tfrt_fallback.tf_tensor,
    %ch2      : !tfrt.chain,
    %ch3      : !tfrt.chain
  )
  duration_secs = 3, max_count = 100000, num_warmup_runs = 10
  {
    %result = tfrt_fallback_async.executeop key(2) cost(100) device("/CPU:0")
       "tf.Rsqrt"(%const) { T = f32 } : 1
    tfrt.return %result : !tfrt_fallback.tf_tensor
  }

  tfrt.return
}

// CHECK: --- Running 'BM_fallback_rsqrt_tanh_f32'
func.func @BM_fallback_rsqrt_tanh_f32() {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(3) device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
     } num_args(0)

  %ch2 = tfrt_fallback_async.createop(%ch1) key(4) device("/CPU:0")
    "tf.Rsqrt"() { T = f32 } num_args(1)

  %ch3 = tfrt_fallback_async.createop(%ch2) key(5) device("/CPU:0")
    "tf.Tanh"() { T = f32 } num_args(1)

  %ch4, %const = tfrt_fallback_async.executeop.seq(%ch3) key(3) cost(100)
                                                         device("/CPU:0")
    "tf.Const"() {
       dtype = f32, value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
     } : 1

  tfrt_test.benchmark "BM_fallback_rsqrt_tanh_f32"(
    %const    : !tfrt_fallback.tf_tensor,
    %ch4      : !tfrt.chain
  )
  duration_secs = 3, max_count = 100000, num_warmup_runs = 10
  {
    %result0 = tfrt_fallback_async.executeop key(4) cost(100) device("/CPU:0")
       "tf.Rsqrt"(%const) { T = f32 } : 1
    %result1 = tfrt_fallback_async.executeop key(5) cost(100) device("/CPU:0")
       "tf.Tanh"(%result0) { T = f32 } : 1
    tfrt.return %result1 : !tfrt_fallback.tf_tensor
  }

  tfrt.return
}

