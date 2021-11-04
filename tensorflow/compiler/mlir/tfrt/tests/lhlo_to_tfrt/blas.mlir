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
func @gemm(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>, %output:memref<5x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[M:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[N:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[K:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 5.000000e-01
  // CHECK: [[LDA:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[LDB:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[BETA:%[0-9]+]] = tfrt.constant.f32 0.000000e+00
  // CHECK: [[LDC:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[ALGO:%[0-9]+]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.blas.create %arg1

  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.blas.gemm [[HANDLE]],
  // CHECK-SAME: CUBLAS_OP_N, CUBLAS_OP_N, [[M]], [[N]], [[K]], [[ALPHA]],
  // CHECK-SAME: %arg3, CUDA_R_32F, [[LDA]],
  // CHECK-SAME: %arg2, CUDA_R_32F, [[LDB]], [[BETA]],
  // CHECK-SAME: %arg4, CUDA_R_32F, [[LDC]],
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
    batch_size = 1,
    lhs_stride = 20,
    rhs_stride = 20
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()

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
func @gemm_batch(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>, %output:memref<5x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[M:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[N:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[K:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 5.000000e-01
  // CHECK: [[LDA:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[LDB:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[BETA:%[0-9]+]] = tfrt.constant.f32 0.000000e+00
  // CHECK: [[LDC:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[ALGO:%[0-9]+]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.blas.create %arg1
  // CHECK: [[STRIDEA:%[0-9]+]] = tfrt.constant.i64 20
  // CHECK: [[STRIDEB:%[0-9]+]] = tfrt.constant.i64 20
  // CHECK: [[STRIDEC:%[0-9]+]] = tfrt.constant.i64 25
  // CHECK: [[BATCH:%[0-9]+]] = tfrt.constant.i32 42

  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.blas.gemm.batch [[HANDLE]],
  // CHECK-SAME: CUBLAS_OP_N, CUBLAS_OP_N, [[M]], [[N]], [[K]], [[ALPHA]],
  // CHECK-SAME: %arg3, CUDA_R_32F, [[LDA]], [[STRIDEA]],
  // CHECK-SAME: %arg2, CUDA_R_32F, [[LDB]], [[STRIDEB]], [[BETA]],
  // CHECK-SAME: %arg4, CUDA_R_32F, [[LDC]], [[STRIDEC]], [[BATCH]],
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
    batch_size = 42,
    lhs_stride = 20,
    rhs_stride = 20
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @gemm_bias(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @gemm_bias(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>,
                %bias: memref<5x5xf32>, %output:memref<5x5xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN0:%[0-9]+]] = tfrt_gpu.mem.copy %arg5, %arg4, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, !tfrt_gpu.buffer

  // CHECK: [[M:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[N:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[K:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 5.000000e-01
  // CHECK: [[LDA:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[LDB:%[0-9]+]] = tfrt.constant.i32 4
  // CHECK: [[BETA:%[0-9]+]] = tfrt.constant.f32 1.000000e+00
  // CHECK: [[LDC:%[0-9]+]] = tfrt.constant.i32 5
  // CHECK: [[ALGO:%[0-9]+]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.blas.create %arg1

  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.blas.gemm [[HANDLE]],
  // CHECK-SAME: CUBLAS_OP_N, CUBLAS_OP_N, [[M]], [[N]], [[K]], [[ALPHA]],
  // CHECK-SAME: %arg3, CUDA_R_32F, [[LDA]],
  // CHECK-SAME: %arg2, CUDA_R_32F, [[LDB]], [[BETA]],
  // CHECK-SAME: %arg5, CUDA_R_32F, [[LDC]],
  // CHECK-SAME: CUBLAS_COMPUTE_32F, [[ALGO]], [[CHAIN0]]

  "lmhlo_gpu.gemm_bias"(%lhs, %rhs, %bias, %output) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    alpha_real = 0.5,
    alpha_imag = 0.0,
    beta = 1.0,
    batch_size = 1,
    lhs_stride = 20,
    rhs_stride = 20
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>, memref<5x5xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN1]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @triangular_solve(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @triangular_solve(%a: memref<2x2xf32>, %b: memref<2x2xf32>, %output: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN0:%[0-9]+]] = tfrt_gpu.mem.copy %arg4, %arg3, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.blas.create %arg1
  // CHECK: [[M:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK: [[N:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f32 1.000000e+00
  // CHECK: [[HEIGHT_A:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK: [[HEIGHT_B:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK: [[BATCH_COUNT:%[0-9]+]] = tfrt.constant.i32 1
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.blas.trsm.batch [[HANDLE]],
  // CHECK-SAME: CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
  // CHECK-SAME: CUBLAS_DIAG_UNIT, [[M]], [[N]], CUDA_R_32F, [[ALPHA]],
  // CHECK-SAME: %arg2, [[HEIGHT_A]], %arg4, [[HEIGHT_B]], [[BATCH_COUNT]],
  // CHECK-SAME: [[CHAIN0]]

  "lmhlo.triangular_solve"(%a, %b, %output) {
      layout_a = dense<[0, 1]> : tensor<2xindex>,
      layout_b = dense<[0, 1]> : tensor<2xindex>,
      layout_output = dense<[0, 1]> : tensor<2xindex>,
      left_side = true, lower = true, transpose_a = "NO_TRANSPOSE",
      unit_diagonal = true
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN1]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
