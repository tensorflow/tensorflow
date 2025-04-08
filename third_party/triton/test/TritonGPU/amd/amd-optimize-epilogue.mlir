// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-epilogue | FileCheck %s

// CHECK-LABEL: one_op_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @one_op_in_chain(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: two_ops_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_ops_in_chain(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = arith.truncf %2 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %4, %3 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
