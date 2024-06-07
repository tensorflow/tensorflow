// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma0}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func @sparse_dot(%A: tensor<64x32xf16, #blocked0>, %B: tensor<64x64xf16, #blocked0>, %meta: tensor<64x4xi16, #blocked0>) {
    %A_alloc = triton_gpu.local_alloc %A {allocation.offset = 0 : i32} : (tensor<64x32xf16, #blocked0>) -> !tt.memdesc<64x32xf16, #shared0, #triton_gpu.shared_memory>
    %B_alloc = triton_gpu.local_alloc %B {allocation.offset = 4096 : i32} : (tensor<64x64xf16, #blocked0>) -> !tt.memdesc<64x64xf16, #shared0, #triton_gpu.shared_memory>
    // CHECK-COUNT-2: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_alloc = triton_gpu.local_alloc %meta {allocation.offset = 12288 : i32} : (tensor<64x4xi16, #blocked0>) -> !tt.memdesc<64x4xi16, #shared0, #triton_gpu.shared_memory>
    %meta_reg = triton_gpu.local_load %meta_alloc : !tt.memdesc<64x4xi16, #shared0, #triton_gpu.shared_memory> -> tensor<64x4xi16, #dot_meta_enc>
    // CHECK: nvgpu.wgmma_fence
    // CHECK-COUNT-2: nvgpu.wgmma_sp %[[A:.*]] meta %[[M:.*]], %[[B:.*]], %[[C:.*]] {
    // CHECK-DAG: layoutA = 0 : i32
    // CHECK-DAG: layoutB = 0 : i32
    // CHECK-DAG: m = 64 : i32
    // CHECK-DAG: n = 64 : i32
    // CHECK-DAG: k = 32 : i32
    // CHECK: nvgpu.wgmma_commit_group
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma0>
    %D = triton_gpu.sparse_dot %A_alloc, %B_alloc, %acc, %meta_reg : !tt.memdesc<64x32xf16, #shared0, #triton_gpu.shared_memory> meta tensor<64x4xi16, #dot_meta_enc> * !tt.memdesc<64x64xf16, #shared0, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma0>
    tt.return
  }
}
