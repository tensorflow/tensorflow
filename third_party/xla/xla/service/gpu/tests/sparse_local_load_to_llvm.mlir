// RUN: xla-opt %s -split-input-file --sparse-local-load-to-llvm | FileCheck %s

#shared = #ttg.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#dot_meta_enc = #triton_xla.sparse_dot_meta<{parent=#mma}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: sparse_local_load_ampere
  tt.func @sparse_local_load_ampere(%A_alloc: !ttg.memdesc<32x32xf16, #shared, #ttg.shared_memory>,
                      %B_alloc: !ttg.memdesc<64x32xf16, #shared, #ttg.shared_memory>,
                      %meta_alloc: !ttg.memdesc<32x4xi16, #shared, #ttg.shared_memory>) {
    // A_dot and B_dot local loads shouldn not match with -sparse-local-load-to-llvm
    // CHECK-COUNT-2: ttg.local_load
    %A_dot = ttg.local_load %A_alloc : !ttg.memdesc<32x32xf16, #shared, #ttg.shared_memory> -> tensor<32x32xf16, #dot_operand_a>
    %B_dot = ttg.local_load %B_alloc : !ttg.memdesc<64x32xf16, #shared, #ttg.shared_memory> -> tensor<64x32xf16, #dot_operand_b>
    // CHECK-COUNT-4: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_reg = ttg.local_load %meta_alloc : !ttg.memdesc<32x4xi16, #shared, #ttg.shared_memory> -> tensor<32x4xi16, #dot_meta_enc>
    tt.return
  }
}

// -----

#shared = #ttg.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>
#dot_meta_enc = #triton_xla.sparse_dot_meta<{parent=#mma}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: sparse_local_load_hopper
  tt.func @sparse_local_load_hopper(%meta_alloc: !ttg.memdesc<64x4xi16, #shared, #ttg.shared_memory>) {
    // CHECK-COUNT-2: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_reg = ttg.local_load %meta_alloc : !ttg.memdesc<64x4xi16, #shared, #ttg.shared_memory> -> tensor<64x4xi16, #dot_meta_enc>
   tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: skip_pass_if_no_sparse_loads
  tt.func @skip_pass_if_no_sparse_loads(%A: tensor<32x64xf16, #blocked>, %B: tensor<64x32xf16, #blocked>) {
    // CHECK-NOT: llvm
    // CHECK-NOT: barrier
    %A_alloc = ttg.local_alloc %A {allocation.offset = 0 : i32} : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory>
    %A_dot = ttg.local_load %A_alloc : !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory> -> tensor<32x64xf16, #dot_operand_a>
    %B_alloc = ttg.local_alloc %B {allocation.offset = 2048 : i32} : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #ttg.shared_memory>
    %B_dot = ttg.local_load %B_alloc : !ttg.memdesc<64x32xf16, #shared, #ttg.shared_memory> -> tensor<64x32xf16, #dot_operand_b>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %D = tt.dot %A_dot, %B_dot, %acc : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

