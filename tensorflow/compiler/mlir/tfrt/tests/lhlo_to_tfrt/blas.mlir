// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @gemm(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @gemm(%lhs: memref<3x4xf32>, %rhs: memref<4x5xf32>, %output:memref<3x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK-DAG: [[M:%[0-9]+]] = tfrt.constant.i32 3
  // CHECK-DAG: [[N:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK-DAG: [[K:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK-DAG: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 5.000000e-01
  // CHECK-DAG: [[BETA:%[0-9]+]] = tfrt.constant.f32 0.000000e+00
  // CHECK: [[ALGO:%[0-9]+]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt.once @tfrt_gpu.blas.create{{.*}}([[CONTEXT]])

  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.blas.gemm [[HANDLE]], %arg1
  // CHECK-SAME: CUBLAS_OP_N, CUBLAS_OP_N, [[N]], [[M]], [[K]], [[ALPHA]],
  // CHECK-SAME: %arg3, CUDA_R_32F, [[N]],
  // CHECK-SAME: %arg2, CUDA_R_32F, [[K]], [[BETA]],
  // CHECK-SAME: %arg4, CUDA_R_32F, [[N]],
  // CHECK-SAME: CUBLAS_COMPUTE_32F, [[ALGO]], %arg0

  "lmhlo_gpu.gemm"(%lhs, %rhs, %output) {
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
    lhs_stride = 12,
    rhs_stride = 20
  } : (memref<3x4xf32>, memref<4x5xf32>, memref<3x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @gemm_batch(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @gemm_batch(%lhs: memref<42x3x4xf32>, %rhs: memref<4x5xf32>, %output:memref<42x3x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK-DAG: [[M:%[0-9]+]] = tfrt.constant.i32 3
  // CHECK-DAG: [[N:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK-DAG: [[K:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK-DAG: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 5.000000e-01
  // CHECK-DAG: [[BETA:%[0-9]+]] = tfrt.constant.f32 0.000000e+00
  // CHECK: [[ALGO:%[0-9]+]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt.once @tfrt_gpu.blas.create{{.*}}([[CONTEXT]])
  // CHECK-DAG: [[STRIDEA:%[0-9]+]] = tfrt.constant.i64 12
  // CHECK-DAG: [[STRIDEB:%[0-9]+]] = tfrt.constant.i64 0
  // CHECK-DAG: [[STRIDEC:%[0-9]+]] = tfrt.constant.i64 15
  // CHECK-DAG: [[BATCH:%[0-9]+]] = tfrt.constant.i32 42

  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.blas.gemm.batch [[HANDLE]], %arg1,
  // CHECK-SAME: CUBLAS_OP_N, CUBLAS_OP_N, [[N]], [[M]], [[K]], [[ALPHA]],
  // CHECK-SAME: %arg3, CUDA_R_32F, [[N]], [[STRIDEB]],
  // CHECK-SAME: %arg2, CUDA_R_32F, [[K]], [[STRIDEA]], [[BETA]],
  // CHECK-SAME: %arg4, CUDA_R_32F, [[N]], [[STRIDEC]], [[BATCH]],
  // CHECK-SAME: CUBLAS_COMPUTE_32F, [[ALGO]], %arg0

  "lmhlo_gpu.gemm"(%lhs, %rhs, %output) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 0.5,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 42,
    lhs_stride = 12,
    rhs_stride = 0
  } : (memref<42x3x4xf32>, memref<4x5xf32>, memref<42x3x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @triangular_solve(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @triangular_solve(%a: memref<2x2xf32>, %b: memref<2x2xf32>, %output: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN0:%[0-9]+]] = tfrt_gpu.mem.copy %arg4, %arg3, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt.once @tfrt_gpu.blas.create{{.*}}([[CONTEXT]])
  // CHECK-DAG: [[C2:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK-DAG: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 1.000000e+00
  // CHECK-DAG: [[BATCH_COUNT:%[0-9]+]] = tfrt.constant.i32 1
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.blas.trsm.batch [[HANDLE]], %arg1,
  // CHECK-SAME: CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  // CHECK-SAME: CUBLAS_DIAG_UNIT, [[C2]], [[C2]], CUDA_R_32F, [[ALPHA]],
  // CHECK-SAME: %arg2, [[C2]], %arg4, [[C2]], [[BATCH_COUNT]],
  // CHECK-SAME: [[CHAIN0]]

  "lmhlo.triangular_solve"(%a, %b, %output) {
      layout_a = dense<[0, 1]> : tensor<2xindex>,
      layout_b = dense<[0, 1]> : tensor<2xindex>,
      layout_output = dense<[0, 1]> : tensor<2xindex>,
      left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>,
      unit_diagonal = true
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN1]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
