// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @fft(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @fft(%input: memref<3x5xf32>, %output: memref<3x5xcomplex<f32>>) {
  // CHECK-NOT: cast

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.fft.create
  // CHECK-SAME: [[CONTEXT]], CUFFT_R2C, 3, [5], [5, 1], [5, 1]
  // CHECK: [[SIZE:%[0-9]+]] = tfrt_gpu.fft.get_workspace_size [[HANDLE]]
  // CHECK: [[ALLOC:%[0-9]+]] = tfrt_gpu.allocator.create [[CONTEXT]]
  // CHECK: [[WORKSPACE:%[0-9]+]] = tfrt_gpu.mem.allocate
  // CHECK-SAME: [[ALLOC]], %arg1, [[SIZE]], %arg0
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.fft.execute %arg1, [[HANDLE]],
  // CHECK-SAME: %arg2, %arg3, [[WORKSPACE]], CUFFT_FORWARD, %arg0
  "lmhlo.fft"(%input, %output) {
    fft_length = dense<5> : tensor<1xi64>,
    fft_type = #mhlo<fft_type RFFT>
  } : (memref<3x5xf32>, memref<3x5xcomplex<f32>>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
