// RUN: xla-opt %s --tritongpu-optimize-dot-operands

// Verify fix for b/439549903.

!tensor1 = tensor<128x16x2xbf16, #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>>
!tensor2 = tensor<128x32xbf16, #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
!memdesc = !ttg.memdesc<128x32xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.target" = "cuda:100",
  "ttg.threads-per-warp" = 32 : i32
} {
  tt.func @reshape_crash(%arg0: !tensor1) -> !memdesc {
    %0 = tt.reshape %arg0 : !tensor1 -> !tensor2
    %1 = ttg.local_alloc %0 : (!tensor2) -> !memdesc
    tt.return %1 : !memdesc
  }
}
