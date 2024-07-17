// RUN: xla-opt %s -split-input-file --sparse-local-load-to-llvm | FileCheck %s

#shared = #triton_gpu.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_local_load_ampere
  tt.func @sparse_local_load_ampere(%A_alloc: !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory>, 
                      %B_alloc: !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory>, 
                      %meta_alloc: !tt.memdesc<32x4xi16, #shared, #triton_gpu.shared_memory>) {
    // A_dot and B_dot local loads shouldn not match with -sparse-local-load-to-llvm
    // CHECK-COUNT-2: triton_gpu.local_load
    %A_dot = triton_gpu.local_load %A_alloc : !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<32x32xf16, #dot_operand_a>
    %B_dot = triton_gpu.local_load %B_alloc : !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x32xf16, #dot_operand_b>
    // CHECK-COUNT-4: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_reg = triton_gpu.local_load %meta_alloc : !tt.memdesc<32x4xi16, #shared, #triton_gpu.shared_memory> -> tensor<32x4xi16, #dot_meta_enc>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_local_load_hopper
  tt.func @sparse_local_load_hopper(%meta_alloc: !tt.memdesc<64x4xi16, #shared, #triton_gpu.shared_memory>) {
    // CHECK-COUNT-2: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_reg = triton_gpu.local_load %meta_alloc : !tt.memdesc<64x4xi16, #shared, #triton_gpu.shared_memory> -> tensor<64x4xi16, #dot_meta_enc>
   tt.return
  }
}

