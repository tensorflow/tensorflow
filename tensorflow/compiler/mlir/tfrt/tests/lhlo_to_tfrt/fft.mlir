// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @fft(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @fft(%input: memref<3x9xf32>, %output: memref<3x5xcomplex<f32>>) {
  // CHECK-NOT: cast
  // CHECK-NOT: as

  // CHECK: [[HANDLE:%[0-9+]]] = tfrt_gpu.fft_create_handle %arg1
  // CHECK: [[CHAIN0:%[0-9+]]] = tfrt_gpu.fft_create_plan %arg1, [[HANDLE]], 7, dense<9> : tensor<1xi64>, [3, 9], [3, 5], %arg0
  // CHECK: [[CHAIN1:%[0-9+]]] = tfrt_gpu.fft_exec %arg1, [[HANDLE]], %arg2, %arg3, dense<9> : tensor<1xi64>, 7, [3, 9], [3, 5], [[CHAIN0]]

  "lmhlo.fft"(%input, %output) {fft_length = dense<9> : tensor<1xi64>, fft_type = "RFFT"} : (memref<3x9xf32>, memref<3x5xcomplex<f32>>) -> ()


  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN1]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
