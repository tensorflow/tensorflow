// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @view(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @view(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>, %output:memref<100xi8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute
  // CHECK-NOT: memref.view

  %c0 = arith.constant 0 : index
  %view = memref.view %output[%c0][] : memref<100xi8> to memref<5x5xf32>

  // CHECK: tfrt_gpu.blas.gemm
  "lmhlo_gpu.gemm"(%lhs, %rhs, %view) {
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
    lhs_stride = 20,
    rhs_stride = 20
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}} : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @reinterpret_cast(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @reinterpret_cast(%lhs: memref<5x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 2)>>, %rhs: memref<4x5xf32>, %output:memref<5x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute
  // CHECK-NOT: memref.reinterpret_cast

  %cast = memref.reinterpret_cast %lhs to offset: [0], sizes: [5, 4], strides: [4, 1] : memref<5x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 2)>> to memref<5x4xf32>

  // CHECK: tfrt_gpu.blas.gemm
  "lmhlo_gpu.gemm"(%cast, %rhs, %output) {
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
    lhs_stride = 20,
    rhs_stride = 20
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}} : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @two_ops(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @two_ops(%memref: memref<4x4xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: tfrt.constant.f32 3.14159274
  // CHECK: tfrt_gpu.blas.gemm
  "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 3.14159274,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 1,
    lhs_stride = 16,
    rhs_stride = 16
  } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK: tfrt.constant.f32 2.71828175
  // CHECK: tfrt_gpu.blas.gemm
  "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 2.71828175,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 1,
    lhs_stride = 16,
    rhs_stride = 16
  } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}} : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @return(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> (!tfrt.chain, !tfrt_gpu.buffer)
func.func @return(%memref: memref<4x4xf32>) -> memref<4x4xf32> {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: tfrt_gpu.blas.gemm
  "lmhlo_gpu.gemm"(%memref, %memref, %memref) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 1.0,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 1,
    lhs_stride = 16,
    rhs_stride = 16
  } : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}}, %arg2 : !tfrt.chain, !tfrt_gpu.buffer
  func.return %memref : memref<4x4xf32>
}
