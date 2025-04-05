// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

// CHECK-LABEL: llvm.func @sink_setprio
// CHECK: rocdl.mfma
// CHECK-NOT: rocdl.mfma
// CHECK: rocdl.s.setprio 1
// CHECK-COUNT-15: rocdl.mfma
// CHECK-NOT: rocdl.mfma
// CHECK: rocdl.s.setprio 0

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_setprio(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    rocdl.s.setprio 1
    %dot = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    rocdl.s.setprio 0
    tt.return
  }
}
