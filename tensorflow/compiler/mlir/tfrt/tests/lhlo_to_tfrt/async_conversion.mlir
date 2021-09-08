// RUN: lhlo-tfrt-opt %s              \
// RUN:   -lmhlo-gpu-async-conversion \
// RUN:   -gpu-async-region           \
// RUN:   -async-gpu-tfrt-conversion  \
// RUN: | FileCheck %s

// CHECK:      func @async(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @async(%memref: memref<4x4xf32>) {
  // CHECK-NOT: cast

  // CHECK: %[[a0:.*]], %[[t0:.*]] = async.execute
  // CHECK-SAME: -> !async.value<!tfrt_gpu.event>
  %a0 = async.execute () {
    // CHECK: %[[e0:.*]] = tfrt_gpu.event.create
    // CHECK: %[[ch0:.*]] = tfrt_gpu.event.record %[[e0]], %arg1
    // CHECK: %[[s0:.*]] = tfrt_gpu.stream.create
    // CHECK: %[[ch1:.*]] = tfrt_gpu.stream.wait %[[s0]], %[[e0]]
    // CHECK: %[[h0:.*]] = tfrt_gpu.blas.create %[[s0]]
    // CHECK: %[[ch2:.*]] = tfrt_gpu.blas.gemm %[[h0]]
    // CHECK-SAME: %[[ch1]]
    "lmhlo_gpu.gemm"(%memref, %memref, %memref) { dot_dimension_numbers = {
         lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
         rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
         lhs_contracting_dimensions = dense<[1]> : tensor<1xi64>,
         rhs_contracting_dimensions = dense<[0]> : tensor<1xi64>},
         alpha_real = 0.5,
         alpha_imag = 0.0,
         batch_size = 1,
         lhs_stride = 16,
         rhs_stride = 16}
      : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    // CHECK: %[[e1:.*]] = tfrt_gpu.event.create
    // CHECK: %[[ch3:.*]] = tfrt_gpu.event.record %[[e1]], %[[s0]], %[[ch2]]
    // CHECK: async.yield %[[e1]] : !tfrt_gpu.event
    async.yield
  }

  // CHECK: %[[a1:.*]], %[[t1:.*]] = async.execute [%[[a0]]]
  // CHECK-SAME: (%[[t0]] as %[[e2:.*]]:
  // CHECK-SAME: !async.value<!tfrt_gpu.event>) -> !async.value<!tfrt_gpu.event>
  %a1 = async.execute [%a0] () {
    // CHECK: %[[s1:.*]] = tfrt_gpu.stream.create
    // CHECK: %[[ch4:.*]] = tfrt_gpu.stream.wait %[[s1]], %[[e2]]
    // CHECK: %[[h1:.*]] = tfrt_gpu.blas.create %[[s1]]
    // CHECK: %[[ch5:.*]] = tfrt_gpu.blas.gemm %[[h1]]
    // CHECK-SAME: %[[ch4]]
    "lmhlo_gpu.gemm"(%memref, %memref, %memref) { dot_dimension_numbers = {
         lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
         rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
         lhs_contracting_dimensions = dense<[1]> : tensor<1xi64>,
         rhs_contracting_dimensions = dense<[0]> : tensor<1xi64>},
         alpha_real = 0.5,
         alpha_imag = 0.0,
         batch_size = 1,
         lhs_stride = 16,
         rhs_stride = 16}
      : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    // CHECK: %[[e3:.*]] = tfrt_gpu.event.create
    // CHECK: %[[ch6:.*]] = tfrt_gpu.event.record %[[e3]], %[[s1]], %[[ch5]]
    // CHECK: async.yield %[[e3]] : !tfrt_gpu.event
    async.yield
  }

  // CHECK: async.await %[[a1]] : !async.token
  // CHECK: %[[e4:.*]] = async.await %[[t1]] : !async.value<!tfrt_gpu.event>
  // CHECK: %[[ch7:.*]] = tfrt_gpu.stream.wait %arg1, %[[e4]]
  async.await %a1 : !async.token

  // CHECK: %[[h2:.*]] = tfrt_gpu.blas.create %arg1
  // CHECK: %[[ch8:.*]] = tfrt_gpu.blas.gemm %[[h2]]
  // CHECK-SAME: %[[ch7]]
  "lmhlo_gpu.gemm"(%memref, %memref, %memref) { dot_dimension_numbers = {
       lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
       rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
       lhs_contracting_dimensions = dense<[1]> : tensor<1xi64>,
       rhs_contracting_dimensions = dense<[0]> : tensor<1xi64>},
       alpha_real = 0.5,
       alpha_imag = 0.0,
       batch_size = 1,
       lhs_stride = 16,
       rhs_stride = 16}
    : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK: tfrt.return %[[ch8]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
