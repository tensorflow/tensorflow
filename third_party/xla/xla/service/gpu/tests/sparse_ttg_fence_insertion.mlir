// RUN: xla-opt %s -split-input-file -triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#lhs = #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>
#rhs = #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func public @sparse_dot_fence(%A: tensor<64x32xf16, #lhs>, %B: tensor<64x64xf16, #rhs>, %meta: tensor<64x4xi16, #blocked>) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = triton_gpu.local_alloc %A : (tensor<64x32xf16, #lhs>) -> !tt.memdesc<64x32xf16, #shared>
    %1 = triton_gpu.local_alloc %B : (tensor<64x64xf16, #rhs>) -> !tt.memdesc<64x64xf16, #shared>
    %2 = triton_gpu.convert_layout %meta : tensor<64x4xi16, #blocked> -> tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #mma}>>
    // CHECK: triton_nvidia_gpu.fence_async_shared
    %3 = triton_xla.sparse_dot %0, %1, %C, %2 : !tt.memdesc<64x32xf16, #shared> meta tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #mma}>> * !tt.memdesc<64x64xf16, #shared> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
