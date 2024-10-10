// TODO(b/350928208): Isolate --sparse-dot-to-llvm pass in this test.
// RUN: xla-opt %s \
// RUN:   --convert-triton-gpu-to-llvm=compute-capability=90 \
// RUN:   --sparse-dot-to-llvm \
// RUN: | FileCheck %s

#shared = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_dot_to_llvm_hopper
  tt.func @sparse_dot_to_llvm_hopper(%A_alloc: !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory>,
                      %B_alloc: !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>,
                      %meta_reg: tensor<64x4xi16, #dot_meta_enc>) {
    // CHECK-NOT: gpu.thread_id
    // CHECK: nvgpu.wgmma_fence
    // CHECK-COUNT-2: nvgpu.wgmma_sp %[[A:.*]] meta %[[M:.*]], %[[B:.*]], %[[C:.*]] {
    // CHECK-DAG: layoutA = 0 : i32
    // CHECK-DAG: layoutB = 0 : i32
    // CHECK-DAG: m = 64 : i32
    // CHECK-DAG: n = 64 : i32
    // CHECK-DAG: k = 32 : i32
    // CHECK: nvgpu.wgmma_commit_group
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %D = triton_xla.sparse_dot %A_alloc, %B_alloc, %acc, %meta_reg : !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory> meta tensor<64x4xi16, #dot_meta_enc> * !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
