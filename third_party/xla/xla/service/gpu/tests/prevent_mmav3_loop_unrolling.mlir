// RUN: xla-opt %s -split-input-file -prevent-mmav3-loop-unrolling | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @add_pragma_nounroll
tt.func @add_pragma_nounroll(%arg0: !ttg.memdesc<64x32xf16, #shared, #smem>, %arg1: !ttg.memdesc<32x32xf16, #shared, #smem>) {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
  %38:1 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg3 = %cst_2) -> (tensor<64x32xf32, #mma>)  : i32 {
    // CHECK: scf.for
    // CHECK-NEXT: tt.elementwise_inline_asm ".pragma \22nounroll\22;"
    // CHECK-SAME: pure = false
    %dot = ttng.warp_group_dot %arg0, %arg1, %arg3 : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x32xf16, #shared, #smem> -> tensor<64x32xf32, #mma>
    scf.yield %dot : tensor<64x32xf32, #mma>
  }
  tt.return
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
// CHECK-LABEL: @do_not_unroll_loops_without_mmav3
tt.func @do_not_unroll_loops_without_mmav3(%arg0: tensor<64x32xf16, #dot_a>, %arg1: tensor<32x32xf16, #dot_b>) {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
  %38:1 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg3 = %cst_2) -> (tensor<64x32xf32, #mma>)  : i32 {
    // CHECK-NOT: tt.elementwise_inline_asm ".pragma \22nounroll\22;"
    %dot = tt.dot %arg0, %arg1, %arg3 : tensor<64x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<64x32xf32, #mma>
    scf.yield %dot : tensor<64x32xf32, #mma>
  }
  tt.return
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @add_pragma_unroll_exactly_once
tt.func @add_pragma_unroll_exactly_once(%arg0: !ttg.memdesc<64x32xf16, #shared, #smem>, %arg1: !ttg.memdesc<32x32xf16, #shared, #smem>) {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
  %38:1 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg3 = %cst_2) -> (tensor<64x32xf32, #mma>)  : i32 {
    // CHECK: scf.for
    // CHECK-COUNT-1: tt.elementwise_inline_asm ".pragma \22nounroll\22;"
    // CHECK-NOT: tt.elementwise_inline_asm ".pragma \22nounroll\22;"
    %dot = ttng.warp_group_dot %arg0, %arg1, %arg3 : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x32xf16, #shared, #smem> -> tensor<64x32xf32, #mma>
    %dot2 = ttng.warp_group_dot %arg0, %arg1, %arg3 : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x32xf16, #shared, #smem> -> tensor<64x32xf32, #mma>
    scf.yield %dot : tensor<64x32xf32, #mma>
  }
  tt.return
}
