// RUN: triton-opt %s -split-input-file -tritonamdgpu-in-thread-transpose | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[$OLD_LAYOUT1:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
// CHECK-DAG: [[$OLD_LAYOUT2:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
// CHECK-DAG: [[$TRANSPOSABLE_LAYOUT1:#.*]] = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-DAG: [[$TRANSPOSABLE_LAYOUT2:#.*]] = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-DAG: [[$LINEAR1:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [1, 0], [2, 0], [4, 0{{]]}}, lane = {{\[\[}}8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 4{{]]}}, warp = {{\[\[}}0, 8], [0, 16], [0, 0{{]]}}, block = []}>
// CHECK-DAG: [[$LINEAR2:#.*]] = #ttg.linear<{register = {{\[\[}}1, 0], [2, 0], [0, 1], [0, 2], [0, 4{{]]}}, lane = {{\[\[}}0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [8, 0{{]]}}, warp = {{\[\[}}16, 0], [0, 0], [0, 0{{]]}}, block = []}>
// CHECK-DAG: [[$SHARED1:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: [[$SHARED2:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>

// CHECK-LABEL: inThreadTranspose_simple

// CHECK-DAG: [[LOAD_VAL1:%.*]] = tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$TRANSPOSABLE_LAYOUT1]]>
// CHECK-DAG: [[LOAD_VAL2:%.*]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$TRANSPOSABLE_LAYOUT2]]>

// CHECK-DAG: [[TMP1_VAL1:%.*]] = ttg.convert_layout [[LOAD_VAL1]] : tensor<256x32xf16, [[$TRANSPOSABLE_LAYOUT1]]> -> tensor<256x32xf16, [[$OLD_LAYOUT1]]>
// CHECK-DAG: [[TMP2_VAL1:%.*]] = ttg.convert_layout [[TMP1_VAL1]] : tensor<256x32xf16, [[$OLD_LAYOUT1]]> -> tensor<256x32xf16, [[$TRANSPOSABLE_LAYOUT1]]>
// CHECK-DAG: [[TRANSPOSED_VAL1:%.*]] = amdgpu.in_thread_transpose [[TMP2_VAL1]] : tensor<256x32xf16, [[$TRANSPOSABLE_LAYOUT1]]> -> tensor<256x32xf16, [[$LINEAR1]]>

// CHECK-DAG: [[TMP1_VAL2:%.*]] = ttg.convert_layout [[LOAD_VAL2]] : tensor<32x128xf16, [[$TRANSPOSABLE_LAYOUT2]]> -> tensor<32x128xf16, [[$OLD_LAYOUT2]]>
// CHECK-DAG: [[TMP2_VAL2:%.*]] = ttg.convert_layout [[TMP1_VAL2]] : tensor<32x128xf16, [[$OLD_LAYOUT2]]> -> tensor<32x128xf16, [[$TRANSPOSABLE_LAYOUT2]]>
// CHECK-DAG: [[TRANSPOSED_VAL2:%.*]] = amdgpu.in_thread_transpose [[TMP2_VAL2]] : tensor<32x128xf16, [[$TRANSPOSABLE_LAYOUT2]]> -> tensor<32x128xf16, [[$LINEAR2]]>

// CHECK-DAG: [[ALLOC1:%.*]] = ttg.local_alloc [[TRANSPOSED_VAL1]] : (tensor<256x32xf16, [[$LINEAR1]]>) -> !ttg.memdesc<256x32xf16, [[$SHARED1]], #smem>
// CHECK-DAG: [[ALLOC2:%.*]] = ttg.local_alloc [[TRANSPOSED_VAL2]] : (tensor<32x128xf16, [[$LINEAR2]]>) -> !ttg.memdesc<32x128xf16, [[$SHARED2]], #smem>
// CHECK-DAG: ttg.local_load [[ALLOC1]] : !ttg.memdesc<256x32xf16, [[$SHARED1]], #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK-DAG: ttg.local_load [[ALLOC2]] : !ttg.memdesc<32x128xf16, [[$SHARED2]], #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  tt.func public @inThreadTranspose_simple(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-DAG: [[$BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-DAG: [[$BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-LABEL: inThreadTranspose_k_fast_neg
// CHECK-DAG: tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$BLOCKED1]]>
// CHECK-DAG: tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$BLOCKED2]]>
  tt.func public @inThreadTranspose_k_fast_neg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-DAG: [[$BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-DAG: [[$BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-LABEL: inThreadTranspose_small_k_neg
// CHECK-DAG: tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$BLOCKED1]]>
// CHECK-DAG: tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$BLOCKED2]]>
  tt.func public @inThreadTranspose_small_k_neg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-DAG: [[$OLD_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: [[$TRANSPOSABLE_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: [[$LINEAR:#.*]] = #ttg.linear<{register = {{\[\[}}1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [32, 0{{]]}}, lane = {{\[\[}}0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$SHARED:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>

// CHECK-LABEL: inThreadTranspose_with_cfg

// CHECK-DAG: ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, [[$SHARED]], #smem, mutable>
// CHECK-DAG: [[LOAD_VAL_preloop:%.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[$TRANSPOSABLE_LAYOUT]]>

// CHECK-DAG: [[TMP1_VAL_preloop:%.*]] = ttg.convert_layout [[LOAD_VAL_preloop]] : tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]> -> tensor<64x64xf16, [[$OLD_LAYOUT]]>
// CHECK-DAG: [[TMP2_VAL_preloop:%.*]] = ttg.convert_layout [[TMP1_VAL_preloop]] : tensor<64x64xf16, [[$OLD_LAYOUT]]> -> tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]>
// CHECK-DAG: [[TRANSPOSED_VAL_preloop:%.*]] = amdgpu.in_thread_transpose [[TMP2_VAL_preloop]] : tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]> -> tensor<64x64xf16, [[$LINEAR]]>

// CHECK-DAG: ttg.local_store [[TRANSPOSED_VAL_preloop]], {{.*}} : tensor<64x64xf16, [[$LINEAR]]> -> !ttg.memdesc<64x64xf16, [[$SHARED]], #smem, mutable>
// CHECK: scf.for
// CHECK-DAG: [[LOAD_VAL_loop:%.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[$TRANSPOSABLE_LAYOUT]]>
// CHECK-DAG: ttg.local_load {{.*}} : !ttg.memdesc<64x64xf16, [[$SHARED]], #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

// CHECK-DAG: [[TMP1_VAL_loop:%.*]] = ttg.convert_layout [[LOAD_VAL_loop]] : tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]> -> tensor<64x64xf16, [[$OLD_LAYOUT]]>
// CHECK-DAG: [[TMP2_VAL_loop:%.*]] = ttg.convert_layout [[TMP1_VAL_loop]] : tensor<64x64xf16, [[$OLD_LAYOUT]]> -> tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]>
// CHECK-DAG: [[TRANSPOSED_VAL_loop:%.*]] = amdgpu.in_thread_transpose [[TMP2_VAL_loop]] : tensor<64x64xf16, [[$TRANSPOSABLE_LAYOUT]]> -> tensor<64x64xf16, [[$LINEAR]]>

// CHECK: ttg.local_store [[TRANSPOSED_VAL_loop]], {{.*}} : tensor<64x64xf16, [[$LINEAR]]> -> !ttg.memdesc<64x64xf16, [[$SHARED]], #smem, mutable>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @inThreadTranspose_with_cfg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %cst_1 = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cst_2 = arith.constant dense<true> : tensor<64x64xi1, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %2 = arith.addi %arg5, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.muli %arg7, %c64_i32 : i32
    %5 = tt.splat %4 : i32 -> tensor<64x64xi32, #blocked>
    %6 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %7 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    %8 = tt.load %0, %cst_1 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %9 = tt.load %1, %cst_1 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %10 = ttg.memdesc_subview %6[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttg.local_store %8, %10 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %11 = ttg.memdesc_subview %7[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    ttg.local_store %9, %11 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    %12 = arith.subi %3, %c1_i32 : i32
    %13:6 = scf.for %arg9 = %c0_i32 to %12 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %0, %arg12 = %1, %arg13 = %c0_i32, %arg14 = %10, %arg15 = %11) -> (tensor<64x64xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
      %21 = tt.addptr %arg11, %cst : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %22 = tt.addptr %arg12, %5 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %23 = tt.load %21 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %24 = ttg.local_load %arg14 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %25 = tt.load %22 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %26 = ttg.local_load %arg15 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %27 = tt.dot %24, %26, %arg10, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %28 = arith.addi %arg13, %c1_i32 : i32
      %29 = arith.cmpi slt, %28, %c1_i32 : i32
      %30 = arith.select %29, %28, %c0_i32 : i32
      %31 = ttg.memdesc_subview %6[%30, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.local_store %23, %31 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %32 = ttg.memdesc_subview %7[%30, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      ttg.local_store %25, %32 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      scf.yield %27, %21, %22, %30, %31, %32 : tensor<64x64xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    }
    %14 = arith.cmpi sge, %3, %c1_i32 : i32
    %15 = ttg.local_load %13#4 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %16 = ttg.local_load %13#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %17 = scf.if %14 -> (tensor<64x64xf32, #mma>) {
      %21 = tt.dot %15, %16, %13#0, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      scf.yield %21 : tensor<64x64xf32, #mma>
    } else {
      scf.yield %13#0 : tensor<64x64xf32, #mma>
    }
    %18 = arith.select %14, %17, %13#0 : tensor<64x64xf32, #mma>
    ttg.local_dealloc %6 : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %7 : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    %19 = arith.truncf %18 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    %20 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #mma>
    tt.store %20, %19, %cst_2 : tensor<64x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: inThreadTranspose_multiple_local_loads

// CHECK: [[LOAD_ADDR:%.*]] = tt.splat
// CHECK: [[IF:%.*]] = scf.if
// CHECK-DAG: [[LOAD_ADDR_CVT1:%.*]] = ttg.convert_layout [[LOAD_ADDR]]
// CHECK-DAG: [[LOAD_VAL1:%.*]] = tt.load [[LOAD_ADDR_CVT1]]
// CHECK-DAG: [[LOAD_VAL1_CVT1:%.*]] = ttg.convert_layout [[LOAD_VAL1]]
// CHECK-DAG: [[LOAD_VAL1_CVT2:%.*]] = ttg.convert_layout [[LOAD_VAL1_CVT1:%.*]]
// CHECK-DAG: [[TRANSPOSED_IN_REG1:%.*]] = amdgpu.in_thread_transpose [[LOAD_VAL1_CVT2]]
// CHECK-DAG: [[LOCAL_ALLOC1:%.*]] = ttg.local_alloc [[TRANSPOSED_IN_REG1]]
// CHECK-DAG: [[LOCAL_LOAD1:%.*]] = ttg.local_load [[LOCAL_ALLOC1]]
// CHECK-DAG: scf.yield [[LOCAL_LOAD1]]
// CHECK: } else {
// CHECK-DAG: [[LOAD_ADDR_CVT2:%.*]] = ttg.convert_layout [[LOAD_ADDR]]
// CHECK-DAG: [[LOAD_VAL2:%.*]] = tt.load [[LOAD_ADDR_CVT2]]
// CHECK-DAG: [[LOAD_VAL2_CVT1:%.*]] = ttg.convert_layout [[LOAD_VAL2]]
// CHECK-DAG: [[LOAD_VAL2_CVT2:%.*]] = ttg.convert_layout [[LOAD_VAL2_CVT1:%.*]]
// CHECK-DAG: [[TRANSPOSED_IN_REG2:%.*]] = amdgpu.in_thread_transpose [[LOAD_VAL2_CVT2]]
// CHECK-DAG: [[LOCAL_ALLOC2:%.*]] = ttg.local_alloc [[TRANSPOSED_IN_REG2]]
// CHECK-DAG: [[LOCAL_LOAD2:%.*]] = ttg.local_load [[LOCAL_ALLOC2]]
// CHECK-DAG: scf.yield [[LOCAL_LOAD2]]
// CHECK: tt.dot {{.*}}, [[IF]]
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

  tt.func public @inThreadTranspose_multiple_local_loads(%arg0: tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i1) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %7 = scf.if %arg2 -> (tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) {
      %1 = tt.load %0 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %3 = ttg.local_alloc %1 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
      %4 = ttg.local_load %3 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %4 : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    } else {
      %2 = tt.load %0 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %5 = ttg.local_alloc %2 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
      %6 = ttg.local_load %5 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %6 : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    }

    %8 = tt.dot %arg0, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}
