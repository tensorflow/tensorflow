// RUN: xla-opt %s | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2],
  CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1],
  instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#dot_meta_enc = #triton_xla.sparse_dot_meta<{parent=#mma}>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_xla_triton_op
  tt.func @sparse_xla_triton_op(%A_dot: tensor<32x32xf16, #dot_operand_a>,
   %B_dot: tensor<64x32xf16, #dot_operand_b>,
   %meta_reg: tensor<32x4xi16, #dot_meta_enc>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK-LABEL: triton_xla.sparse_dot
    %D = triton_xla.sparse_dot %A_dot, %B_dot, %acc, %meta_reg :
      tensor<32x32xf16, #dot_operand_a> meta tensor<32x4xi16,
      #dot_meta_enc> * tensor<64x32xf16, #dot_operand_b>
        -> tensor<32x32xf32, #mma>
    tt.return
  }
}
