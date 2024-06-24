// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=80 | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma0}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func @sparse_dot(%A: tensor<32x32xf16, #blocked0>, %B: tensor<64x32xf16, #blocked0>, %meta: tensor<32x4xi16, #blocked0>) {
    // CHECK-COUNT-2: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    %A_alloc = triton_gpu.local_alloc %A {allocation.offset = 0 : i32} : (tensor<32x32xf16, #blocked0>) -> !tt.memdesc<32x32xf16, #shared0, #triton_gpu.shared_memory>
    %A_dot = triton_gpu.local_load %A_alloc : !tt.memdesc<32x32xf16, #shared0, #triton_gpu.shared_memory> -> tensor<32x32xf16, #dot_operand_a>
    // CHECK-COUNT-4: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
    %B_alloc = triton_gpu.local_alloc %B {allocation.offset = 2048 : i32} : (tensor<64x32xf16, #blocked0>) -> !tt.memdesc<64x32xf16, #shared0, #triton_gpu.shared_memory>
    %B_dot = triton_gpu.local_load %B_alloc : !tt.memdesc<64x32xf16, #shared0, #triton_gpu.shared_memory> -> tensor<64x32xf16, #dot_operand_b>
    // CHECK-COUNT-4: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_alloc = triton_gpu.local_alloc %meta {allocation.offset = 6144 : i32} : (tensor<32x4xi16, #blocked0>) -> !tt.memdesc<32x4xi16, #shared0, #triton_gpu.shared_memory>
    %meta_reg = triton_gpu.local_load %meta_alloc : !tt.memdesc<32x4xi16, #shared0, #triton_gpu.shared_memory> -> tensor<32x4xi16, #dot_meta_enc>
    // CHECK-COUNT-4: mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma0>
    %D = triton_gpu.sparse_dot %A_dot, %B_dot, %acc, %meta_reg : tensor<32x32xf16, #dot_operand_a> meta tensor<32x4xi16, #dot_meta_enc> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #mma0>
    tt.return
  }
}
