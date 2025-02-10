// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// TODO(b/394567780): We can remove this file once upstreamed.

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f8_dot_operand(%f16_inp: tensor<32x32xf16, #dot_operand>) {
    // CHECK-LABEL: @f16_to_f8_dot_operand

    %f8 = tt.fp_to_fp %f16_inp, rounding = rtne : tensor<32x32xf16, #dot_operand> -> tensor<32x32xf8E5M2, #dot_operand>
    tt.return
  }
}
