// RUN: xla-opt %s -split-input-file -triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @sparse_dot_fence(%A: tensor<64x32xf16, #lhs>, %B: !ttg.memdesc<64x64xf16, #shared, #smem>, %meta: tensor<64x4xi16, #blocked>) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = ttg.local_alloc %A : (tensor<64x32xf16, #lhs>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
    %2 = ttg.convert_layout %meta : tensor<64x4xi16, #blocked> -> tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #mma}>>
    // CHECK: ttng.fence_async_shared
    %3 = triton_xla.sparse_dot %0, %B, %C, %2 : !ttg.memdesc<64x32xf16, #shared, #smem> meta tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #mma}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
