// RUN: xla-opt %s --sparse-dot-to-llvm | FileCheck %s

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_dot_to_llvm_ampere
  tt.func @sparse_dot_to_llvm_ampere(%A_dot: tensor<32x32xf16, #dot_operand_a>, %B_dot: tensor<64x32xf16, #dot_operand_b>, %meta_reg: tensor<32x4xi16, #dot_meta_enc>) {
    // CHECK-COUNT-4: mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %D = triton_xla.sparse_dot %A_dot, %B_dot, %acc, %meta_reg : tensor<32x32xf16, #dot_operand_a> meta tensor<32x4xi16, #dot_meta_enc> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #mma>
    tt.return
  }
}
