// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=0' | FileCheck %s --check-prefixes MFMA0,CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=16' | FileCheck %s --check-prefixes MFMA16,CHECK

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_fp8e5m2
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_fp8e5m2(
      %arg0: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK: %[[A0:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK: %[[A1:.+]] = tt.fp_to_fp %[[A0]] : {{.*}} -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK: %[[B0:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: %[[B1:.+]] = tt.fp_to_fp %[[B0]] : tensor<64x256xf8E5M2, {{.*}} -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: tt.dot %[[A1]], %[[B1]]
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Verify that we use FMA when the N dimension is too small for any mma.
// MFMA0-NOT: #ttg.amd_mfma
// MFMA16: #ttg.amd_mfma
// CHECK-LABEL: small_n_size
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @small_n_size(
    %a: tensor<4x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>)
    -> tensor<4x128xf32, #blocked> {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked>
    %result = tt.dot %a, %b, %zero_f32 : tensor<4x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x128xf32, #blocked>
    tt.return %result : tensor<4x128xf32, #blocked>
  }
}

// -----

// MFMA0-NOT: amd_mfma
// MFMA16: amd_mfma
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_small_k
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_small_k(
      %arg0: tensor<128x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<4x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<4x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
