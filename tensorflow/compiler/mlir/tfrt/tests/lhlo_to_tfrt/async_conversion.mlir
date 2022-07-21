// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @async(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @async(%memref: memref<4x4xf32>) {
  // CHECK-NOT: cast
  // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: %[[e0:.*]] = tfrt_gpu.event.create %[[ctx]]
  // CHECK: %[[ch0:.*]] = tfrt_gpu.event.record %[[e0]], %arg1

  // CHECK: %[[t0:.*]]:2 = tfrt_test.do.async %[[ctx]], %[[e0]], %[[ch0]], %arg2
  // CHECK-SAME: : (
  // CHECK-SAME:   !tfrt_gpu.context,
  // CHECK-SAME:   !tfrt_gpu.event,
  // CHECK-SAME:   !tfrt.chain,
  // CHECK-SAME:   !tfrt_gpu.buffer
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   !tfrt.chain,
  // CHECK-SAME:   !tfrt_gpu.event
  // CHECK-SAME: )  {
  %a0 = async.execute () {
    // CHECK: %[[s0:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch1:.*]] = tfrt_gpu.stream.wait %[[s0]], %[[e0]], %[[ch0]]
    // CHECK: %[[h0:.*]] = tfrt.once @tfrt_gpu.blas.create
    // CHECK: %[[ch2:.*]] = tfrt_gpu.blas.gemm %[[h0]], %[[s0]]
    // CHECK-SAME: %[[ch1]]
    "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      alpha_real = 0.5,
      alpha_imag = 0.0,
      beta = 0.0,
      batch_size = 1,
      lhs_stride = 16,
      rhs_stride = 16
    } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    // CHECK: %[[e1:.*]] = tfrt_gpu.event.create
    // CHECK: %[[ch3:.*]] = tfrt_gpu.event.record %[[e1]], %[[s0]], %[[ch2]]
    // CHECK: tfrt.return %[[ch3]], %[[e1]] : !tfrt.chain, !tfrt_gpu.event
    async.yield
  }

  // CHECK: %[[t1:.*]]:2 = tfrt_test.do.async %[[t0]]#0, %[[t0]]#1, %[[ctx]], %arg2
  // CHECK-SAME: : (
  // CHECK-SAME:   !tfrt.chain,
  // CHECK-SAME:   !tfrt_gpu.event,
  // CHECK-SAME:   !tfrt_gpu.context,
  // CHECK-SAME:   !tfrt_gpu.buffer
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   !tfrt.chain,
  // CHECK-SAME:   !tfrt_gpu.event
  // CHECK-SAME: )  {
  %a1 = async.execute [%a0] () {
    // CHECK: %[[s1:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch4:.*]] = tfrt_gpu.stream.wait %[[s1]], %[[t0]]#1, %[[t0]]#0
    // CHECK: %[[h1:.*]] = tfrt.once @tfrt_gpu.blas.create
    // CHECK: %[[ch5:.*]] = tfrt_gpu.blas.gemm %[[h1]], %[[s1]]
    // CHECK-SAME: %[[ch4]]
    "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      alpha_real = 0.5,
      alpha_imag = 0.0,
      beta = 0.0,
      batch_size = 1,
      lhs_stride = 16,
      rhs_stride = 16
    } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    // CHECK: %[[e3:.*]] = tfrt_gpu.event.create
    // CHECK: %[[ch6:.*]] = tfrt_gpu.event.record %[[e3]], %[[s1]], %[[ch5]]
    // CHECK: tfrt.return %[[ch6]], %[[e3]] : !tfrt.chain, !tfrt_gpu.event
    async.yield
  }

  // CHECK: %[[ch7:.*]] = tfrt_gpu.stream.wait %arg1, %[[t1]]#1, %[[t1]]#0
  async.await %a1 : !async.token

  // CHECK: %[[h2:.*]] = tfrt.once @tfrt_gpu.blas.create
  // CHECK: %[[ch8:.*]] = tfrt_gpu.blas.gemm %[[h2]], %arg1
  // CHECK-SAME: %[[ch7]]
  "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 0.5,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 1,
    lhs_stride = 16,
    rhs_stride = 16
  } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK: tfrt.return %[[ch8]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
