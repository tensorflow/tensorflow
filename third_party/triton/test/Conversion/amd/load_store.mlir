// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // Load 8 elements from A with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    // Load 8 elements from B with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: global_store_mfma_vec16
  tt.func public @global_store_mfma_vec16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<32x32xf32, #mma>
    %2 = arith.truncf %1 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %c32_i32 = arith.constant 32 : i32
    %100 = tt.get_program_id x : i32
    %101 = arith.muli %100, %c32_i32 : i32
    %102 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %300 = tt.expand_dims %102 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xi32, #mma>
    %200 = tt.broadcast %300 : tensor<1x32xi32, #mma> -> tensor<32x32xi32, #mma>
    %103 = tt.splat %101 : i32 -> tensor<32x32xi32, #mma>
    %104 = arith.addi %103, %200 : tensor<32x32xi32, #mma>
    %105 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #mma>
    %106 = tt.addptr %105, %104 : tensor<32x32x!tt.ptr<f16>, #mma>, tensor<32x32xi32, #mma>
    // Store 16 elements with four vectorized store instruction
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<4xf16>, !llvm.ptr<1>
    tt.store %106, %2 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}
