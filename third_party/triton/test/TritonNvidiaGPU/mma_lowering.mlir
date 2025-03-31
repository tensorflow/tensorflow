// RUN: triton-opt %s -split-input-file --triton-nvidia-mma-lowering | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: gen5_mma_scaled_shmem_to_tmem
  tt.func public @gen5_mma_scaled_shmem_to_tmem(
    %A_sh: !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>,
    %B_sh: !ttg.memdesc<256x64xf8E5M2, #shared, #ttg.shared_memory>,
    %C_tmem: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    %A_scale_sh: !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>,
    %B_scale_sh: !ttg.memdesc<1x2x16x4x4xi8, #shared1, #smem>,
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {

    %true = arith.constant true
    // Verify that the scale in tmem has the shape of (LHS) BlockM x BlockK / 32, (RHS) BlockN x BlockK / 32
    // CHECK: %[[A_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[A_SC_TMEM]]
    // CHECK: %[[B_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<64x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[B_SC_TMEM]]
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, %[[A_SC_TMEM]], %[[B_SC_TMEM]]
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %C_tmem, %A_scale_sh, %B_scale_sh, %true, %true lhs = e5m2 rhs = e5m2, %barrier : (!ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<256x64xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x16x4x4xi8, #shared1, #smem>, i1, i1, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: gen5_mma_scaled_shmem_to_tmem
  tt.func public @gen5_mma_scaled_shmem_to_tmem(
    %A_sh: !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
    %B_sh: !ttg.memdesc<256x64xi8, #shared, #ttg.shared_memory>,
    %C_tmem: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    %A_scale_sh: !ttg.memdesc<1x2x32x4x4xf8E4M3FN, #shared1, #smem>,
    %B_scale_sh: !ttg.memdesc<1x2x16x4x4xf8E4M3FN, #shared1, #smem>,
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {

    %true = arith.constant true
    // Verify that the scale in tmem has the shape of (LHS) BlockM x BlockK / 32, (RHS) BlockN x BlockK / 32
    // CHECK: %[[A_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[A_SC_TMEM]]
    // CHECK: %[[B_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<64x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[B_SC_TMEM]]
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, %[[A_SC_TMEM]], %[[B_SC_TMEM]]
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %C_tmem, %A_scale_sh, %B_scale_sh, %true, %true lhs = e2m1 rhs = e2m1, %barrier : (!ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<256x64xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<1x2x16x4x4xf8E4M3FN, #shared1, #smem>, i1, i1, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) -> ()
    tt.return
  }
}
