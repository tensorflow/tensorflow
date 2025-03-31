// RUN: triton-opt %s  -split-input-file --convert-triton-amdgpu-to-llvm='arch=gfx950' | FileCheck %s

// CHECK-LABEL:mfma_16x16x32_f16

#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_f16(%arg0: tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_16x16x32_bf16

#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf16(%arg0: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_f16

#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_f16(%arg0: tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}


// -----

// CHECK-LABEL:mfma_32x32x16_bf16

#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf16(%arg0: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}
