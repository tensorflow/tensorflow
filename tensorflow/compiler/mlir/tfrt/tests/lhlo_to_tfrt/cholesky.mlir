// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @cholesky(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @cholesky(%input: memref<2x2xf32>, %output: memref<2x2xf32>, %scratch: memref<2x2xf32>, %info: memref<2x2xi32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN0:%[0-9]+]] = tfrt_gpu.mem.copy %arg3, %arg2, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt.once @tfrt_gpu.solver.create{{.*}}([[CONTEXT]])
  // CHECK-DAG: [[N:%[0-9]+]] = tfrt.constant.i32 2
  // CHECK-DAG: [[BATCH_SIZE:%[0-9]+]] = tfrt.constant.i32 1
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.solver.potrf.batch [[HANDLE]], %arg1,
  // CHECK-SAME: CUBLAS_FILL_MODE_LOWER, [[N]], CUDA_R_32F, %arg3, [[N]], %arg5,
  // CHECK-SAME: [[BATCH_SIZE]], [[CHAIN0]]

  "lmhlo_gpu.cholesky"(%input, %output, %scratch, %info) {
      is_lower = true
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xi32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN1]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
