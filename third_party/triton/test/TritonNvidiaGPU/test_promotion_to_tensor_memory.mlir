// RUN: env ENABLE_LHS_TO_TMEM=1 triton-opt %s -split-input-file -tritongpu-promote-lhs-to-tmem | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
// Incompatible access layout for tmem; tmem access requires one thread per datapath
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @no_tmem_promotion
  tt.func public @no_tmem_promotion(
    %lhs: tensor<128x32xf32, #blocked1>,
    %rhs: tensor<32x256xf32, #blocked2>
  ) {
    %true = arith.constant true
    %cst = arith.constant dense<0.0> : tensor<128x256xf32, #blocked>
    // CHECK: ttng.tmem_alloc %[[CST:.*]] : (tensor<128x256xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x256xf32, #tmem
    %tmem = ttng.tmem_alloc %cst :
      (tensor<128x256xf32, #blocked>) ->
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NOT: ttng.tmem_alloc %[[ARG0:.*]] : (tensor<128x32xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x32xf32, #[[TMEM:tmem[0-9]*]]
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x32xf32, #blocked1>) -> !ttg.memdesc<128x32xf32, #shared, #ttg.shared_memory>
    %rhs_shared = ttg.local_alloc %rhs : (tensor<32x256xf32, #blocked2>) -> !ttg.memdesc<32x256xf32, #shared1, #ttg.shared_memory>

    ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %tmem, %true, %true :
      (!ttg.memdesc<128x32xf32, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x256xf32, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1) -> ()

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
// Compatible layout for tmem access
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @promote_lhs_to_tmem
  tt.func public @promote_lhs_to_tmem(
    %lhs: tensor<128x32xf32, #blocked3>,
    %rhs: tensor<32x256xf32, #blocked2>
  ) {
    %true = arith.constant true
    %cst = arith.constant dense<0.0> : tensor<128x256xf32, #blocked>
    // CHECK: ttng.tmem_alloc %[[CST:.*]] : (tensor<128x256xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x256xf32, #tmem
    %tmem = ttng.tmem_alloc %cst :
      (tensor<128x256xf32, #blocked>) ->
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %[[ARG0:.*]] : (tensor<128x32xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x32xf32, #[[TMEM:tmem[0-9]*]]
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x32xf32, #blocked3>) -> !ttg.memdesc<128x32xf32, #shared, #ttg.shared_memory>
    %rhs_shared = ttg.local_alloc %rhs : (tensor<32x256xf32, #blocked2>) -> !ttg.memdesc<32x256xf32, #shared1, #ttg.shared_memory>

    ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %tmem, %true, %true :
      (!ttg.memdesc<128x32xf32, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x256xf32, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1) -> ()

    tt.return
  }
}
