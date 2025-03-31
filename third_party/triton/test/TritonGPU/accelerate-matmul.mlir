// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// CHECK: #[[MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
// CHECK: #[[MMA1:.+]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
// CHECK: #[[MMA2:.+]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: mma_chain_loop
  tt.func public @mma_chain_loop(
   %170: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %153: tensor<128x64x!tt.ptr<f16>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked2>
    // CHECK: scf.for
    // CHECK:   ttng.warp_group_dot {{.*}} -> tensor<128x16xf16, #[[MMA]]>
    // CHECK:   ttng.warp_group_dot {{.*}} -> tensor<128x64xf16, #[[MMA1]]>
    %115 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_0) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %172 = tt.dot %170, %171, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = ttg.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %180 = tt.dot %178, %179, %arg16 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    // CHECK: scf.for
    // CHECK:   ttng.warp_group_dot {{.*}} -> tensor<128x32xf16, #[[MMA2]]>
    // CHECK:   ttng.warp_group_dot {{.*}} -> tensor<128x64xf16, #[[MMA1]]>
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %166 = tt.dot %164, %165, %cst_2 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = ttg.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %174 = tt.dot %172, %173, %arg16 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 32, 16]}>
// CHECK: #mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: chained_dot
  tt.func public @chained_dot_wgmma(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK: ttng.warp_group_dot {{.*}} -> tensor<64x64xf32, #mma>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: ttng.warp_group_dot {{.*}} -> tensor<64x128xf32, #mma1>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:89", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fp8_dot
  tt.func public @fp8_dot(
    %arg0: tensor<64x128xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x64xf32, #blocked> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
  // CHECK: tt.dot {{.*}} : tensor<64x128xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #[[$MMA]], kWidth = 4}>> * tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]], kWidth = 4}>> -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}

// -----

// CHECK-DAG: #[[MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
// CHECK-DAG: #[[MMA1:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1, 1], instrShape = [1, 16, 8]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [0, 1, 2]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [1, 4, 8], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: kernel_
  tt.func public @kernel_() attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x16x16xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked1>
    %0 = ttg.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    %1 = ttg.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
    %2 = ttg.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[MMA]]>
    %3 = tt.dot %0, %1, %2, inputPrecision = tf32 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x16xf32, #blocked1>
    %4 = ttg.convert_layout %3 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x16x16xf32, #blocked2>
    %6 = ttg.convert_layout %5 : tensor<1x16x16xf32, #blocked2> -> tensor<1x16x16xf32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x16x16xf32, #blocked> -> tensor<2x16x16xf32, #blocked>
    %8 = ttg.convert_layout %7 : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
    %9 = ttg.convert_layout %cst : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
    %10 = ttg.convert_layout %cst : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #blocked3>
    // CHECK: tt.dot {{.*}} -> tensor<2x16x16xf32, #[[MMA1]]>
    %11 = tt.dot %8, %9, %10, inputPrecision = tf32 : tensor<2x16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<2x16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<2x16x16xf32, #blocked3>
    %12 = ttg.convert_layout %11 : tensor<2x16x16xf32, #blocked3> -> tensor<2x16x16xf32, #blocked>
    tt.print ": " {hex = false, isSigned = array<i32: 0>} : %12 : tensor<2x16x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, {{.*}}, instrShape = [16, 32, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: check_instrShape_per_warps
  tt.func @check_instrShape_per_warps(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %mask = arith.constant dense<true> : tensor<128x128xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    %result = tt.dot %a, %b, %zero_f32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// Verify that we use mmav2 when the k dim is too small for mmav3.
// CHECK: #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 8], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: small_k_size
  tt.func @small_k_size(
    %a: tensor<128x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>)
    -> tensor<128x128xf32, #blocked> {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result = tt.dot %a, %b, %zero_f32 : tensor<128x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
  // CHECK-DAG: #[[$B:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-DAG: #[[$T:.+]] = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
  // CHECK-LABEL: mmav5
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x64xf16, #{{.*}}>) -> !ttg.memdesc<128x64xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x256xf16, #{{.*}}>) -> !ttg.memdesc<64x256xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x256xf32, #{{.*}}>) -> !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   ttng.tc_gen5_mma %[[A]], %[[B]], %[[ACC]], %[[TRUE]], %[[TRUE]] : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #[[$TMEM]], #ttng.tensor_memory, mutable>, i1, i1) -> ()
  //       CHECK:   %[[R:.+]] = ttng.tmem_load %[[ACC]] : !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable> -> tensor<128x256xf32
  //       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[R]] : tensor<128x256xf32, #[[$T]]> -> tensor<128x256xf32, #[[$B]]>
  //       CHECK:   tt.return %[[CVT]] : tensor<128x256xf32
  tt.func public @mmav5(%a: tensor<128x64xf16, #blocked2>, %b: tensor<64x256xf16, #blocked1>, %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
      %ad = ttg.convert_layout %a : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bd = ttg.convert_layout %b : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %c, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}

// -----

// CHECK: #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 8], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-label: mmav5_fallback_v2_num_warps
  tt.func public @mmav5_fallback_v2_num_warps(%a: tensor<128x64xf16, #blocked2>, %b: tensor<64x256xf16, #blocked1>, %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
      %ad = ttg.convert_layout %a : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bd = ttg.convert_layout %b : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %c, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: mmav5_fp32
  //    CHECK-DAG:   %[[AD:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf32,
  //    CHECK-DAG:   %[[BD:.+]] = ttg.convert_layout %{{.*}} : tensor<64x256xf32,
  //    CHECK-DAG:   %[[D:.*]] = tt.dot %[[AD]], %[[BD]], %{{.*}}
  //    CHECK:   tt.return %[[D]] : tensor<128x256xf32
  tt.func public @mmav5_fp32(%a: tensor<128x64xf32, #blocked2>, %b: tensor<64x256xf32, #blocked1>, %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
      %ad = ttg.convert_layout %a : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bd = ttg.convert_layout %b : tensor<64x256xf32, #blocked1> -> tensor<64x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %c : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, unpacked = true, CTASplitM = 2>
  // CHECK-DAG: #[[$B:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
  // CHECK-DAG: #[[$T:.+]] = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
  // CHECK-LABEL: mmav5
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x64xf16, #{{.*}}>) -> !ttg.memdesc<128x64xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x256xf16, #{{.*}}>) -> !ttg.memdesc<64x256xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x256xf32, #{{.*}}>) -> !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   ttng.tc_gen5_mma %[[A]], %[[B]], %[[ACC]], %[[TRUE]], %[[TRUE]] : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #[[$TMEM]], #ttng.tensor_memory, mutable>, i1, i1) -> ()
  //       CHECK:   %[[R:.+]] = ttng.tmem_load %[[ACC]] : !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable> -> tensor<128x256xf32
  //       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[R]] : tensor<128x256xf32, #[[$T]]> -> tensor<128x256xf32, #[[$B]]>
  //       CHECK:   tt.return %[[CVT]] : tensor<128x256xf32
  tt.func public @mmav5_multi_ctas(%a: tensor<128x64xf16, #blocked2>, %b: tensor<64x256xf16, #blocked1>, %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
      %ad = ttg.convert_layout %a : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bd = ttg.convert_layout %b : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %c, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, unpacked = true, CTASplitM = 2>
  // CHECK-DAG: #[[$B:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
  // CHECK-DAG: #[[$T:.+]] = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
  // CHECK-DAG: #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
  // CHECK-DAG: #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 2], CTAOrder = [1, 0]}>
  // CHECK-LABEL: mmav5
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x64xf16, #{{.*}}>) -> !ttg.memdesc<128x64xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x256xf16, #{{.*}}>) -> !ttg.memdesc<64x256xf16, #{{.*}}, #smem
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x256xf32, #{{.*}}>) -> !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   ttng.tc_gen5_mma %[[A]], %[[B]], %[[ACC]], %[[TRUE]], %[[TRUE]] {two_ctas} : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #[[$TMEM]], #ttng.tensor_memory, mutable>, i1, i1) -> ()
  //       CHECK:   %[[R:.+]] = ttng.tmem_load %[[ACC]] : !ttg.memdesc<128x256xf32, #{{.*}}, #ttng.tensor_memory, mutable> -> tensor<128x256xf32
  //       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[R]] : tensor<128x256xf32, #[[$T]]> -> tensor<128x256xf32, #[[$B]]>
  //       CHECK:   tt.return %[[CVT]] : tensor<128x256xf32
  tt.func public @mmav5_2ctas(%a: tensor<128x64xf16, #blocked2>, %b_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
      %ad = ttg.convert_layout %a : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b = tt.load %b_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %bd = ttg.convert_layout %b : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %c, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
  // CHECK-DAG: #[[$TMEM1:.+]] = #ttng.tensor_memory_scales_encoding
  // CHECK-LABEL: mmav5_block_scaled
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x64xi8, #{{.*}}>) -> !ttg.memdesc<128x64xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x128xi8, #{{.*}}>) -> !ttg.memdesc<64x128xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[SCALEA_LOCAL:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #{{.*}}, #smem>
  //   CHECK:       ttg.local_load %[[SCALEA_LOCAL]] : !ttg.memdesc<128x2xi8, #{{.*}}, #smem> -> tensor<128x2xi8, #{{.*}}>
  //   CHECK-DAG:   %[[SCALEB_LOCAL:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #{{.*}}, #smem>
  //   CHECK:       ttg.local_load %[[SCALEB_LOCAL]] : !ttg.memdesc<128x2xi8, #{{.*}}, #smem> -> tensor<128x2xi8, #{{.*}}>
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #{{.*}}>) -> !ttg.memdesc<128x128xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   %[[SCALEA:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   %[[SCALEB:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   ttng.tc_gen5_mma_scaled %[[A]], %[[B]], %[[ACC]], %[[SCALEA]], %[[SCALEB]], %[[TRUE]], %[[TRUE]] lhs = e4m3 rhs = e4m3
  tt.func public @mmav5_block_scaled(%a: tensor<128x64xi8, #blocked2>, %scale_a_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %scale_a = tt.load %scale_a_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>
    %scale_b = tt.load %scale_b_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}

// -----

// Verify that dot_scaled (mxfp4 x {bf16,fp8}) decomposes to mmav3 if it's bf16, otherwise it fallsback to mmav2
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: #[[LINEAR:.+]] = #ttg.linear<{{.*}}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: dot_scaled
  tt.func @dot_scaled(
    %a: tensor<128x32xi8, #blocked2>,
    %scale: tensor<128x2xi8, #blocked1>,
    %b_bf16: tensor<64x128xbf16, #blocked>
    ) -> tensor<128x128xf32, #blocked> {
    // CHECK: ttg.fp4_to_fp
    // CHECK: ttng.warp_group_dot
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result = tt.dot_scaled %a scale %scale, %b_bf16, %cst lhs = e2m1 rhs = bf16 {fastMath = false} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xbf16, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }

  // Verify that dot_scaled (mxfp4 x fp8) decomposes into mmav3 as well
  // CHECK: dot_scaled_fp8
  tt.func @dot_scaled_fp8(
    %a: tensor<128x32xi8, #blocked2>,
    %scale: tensor<128x2xi8, #blocked1>,
    %b_fp8: tensor<64x128xf8E4M3FN, #blocked>
    ) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: ttg.fp4_to_fp
    // CHECK: ttng.warp_group_dot
    %result = tt.dot_scaled %a scale %scale, %b_fp8, %cst lhs = e2m1 rhs = e4m3 {fastMath = true} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E4M3FN, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

// Mixed dtype matmul with upcasting on the left is transposed and uses MMAv3
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: mixed_dtype_matmul
  tt.func @mixed_dtype_matmul(
    %a: tensor<64x32xf32, #blocked2>,
    %b: tensor<32x64xf8E4M3FN, #blocked1>,
    %c: tensor<64x64xf32, #blocked>
  ) -> tensor<64x64xf32, #blocked> {
    %b_upcast = tt.fp_to_fp %b : tensor<32x64xf8E4M3FN, #blocked1> -> tensor<32x64xf32, #blocked1>
    %a_cvt = ttg.convert_layout %a : tensor<64x32xf32, #blocked2> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_cvt = ttg.convert_layout %b_upcast : tensor<32x64xf32, #blocked1> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: ttng.warp_group_dot
    %d = tt.dot %a_cvt, %b_cvt, %c, inputPrecision = tf32 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$B:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
  // CHECK-DAG: #[[$S:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8, fp4Padded = true}>
  tt.func public @mmav5_block_scaled_mixed_prec(%a: tensor<128x64xi8, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<32x128xi8, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: ttg.local_alloc %arg2 : (tensor<32x128xi8, #[[$B]]>) -> !ttg.memdesc<32x128xi8, #[[$S]], #smem>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<32x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 4, 8, 1, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 1, 2, 3, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[32, 0], [64, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
  // CHECK-DAG: #[[$TMEM1:.+]] = #ttng.tensor_memory_scales_encoding
  // CHECK-LABEL: mmav5_block_scaled_5d_scale
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x128xi8, #{{.*}}>) -> !ttg.memdesc<128x128xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x128xi8, #{{.*}}>) -> !ttg.memdesc<128x128xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[SCALEA_LOCAL:.+]] = ttg.local_alloc
  //   CHECK:       ttg.local_load %[[SCALEA_LOCAL]]
  //   CHECK-DAG:   %[[SCALEB_LOCAL:.+]] = ttg.local_alloc
  //   CHECK:       ttg.local_load %[[SCALEB_LOCAL]]
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #{{.*}}>) -> !ttg.memdesc<128x128xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   %[[SCALEA:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x4xi8, #{{.*}}>) -> !ttg.memdesc<128x4xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   %[[SCALEB:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x4xi8, #{{.*}}>) -> !ttg.memdesc<128x4xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   ttng.tc_gen5_mma_scaled %[[A]], %[[B]], %[[ACC]], %[[SCALEA]], %[[SCALEB]], %[[TRUE]], %[[TRUE]] lhs = e4m3 rhs = e4m3
  tt.func public @mmav5_block_scaled_5d_scale(%a: tensor<128x128xi8, #blocked2>, %scale_a_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>, %b: tensor<128x128xi8, #blocked>, %scale_b_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %scale_a_5d = tt.load %scale_a_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>
    %scale_a_trans = tt.trans %scale_a_5d {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x1x32x4x4xi8, #blocked3> -> tensor<1x4x32x1x4xi8, #blocked4>
    %scale_a = tt.reshape %scale_a_trans : tensor<1x4x32x1x4xi8, #blocked4> -> tensor<128x4xi8, #linear>
    %scale_b_5d = tt.load %scale_b_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>
    %scale_b_trans = tt.trans %scale_b_5d {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x1x32x4x4xi8, #blocked3> -> tensor<1x4x32x1x4xi8, #blocked4>
    %scale_b = tt.reshape %scale_b_trans : tensor<1x4x32x1x4xi8, #blocked4> -> tensor<128x4xi8, #linear>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xi8, #blocked2>, tensor<128x4xi8, #linear> * tensor<128x128xi8, #blocked>, tensor<128x4xi8, #linear> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

tt.func @scalar_load_in_bwd_slice(%arg0: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg1: !tt.tensordesc<tensor<128x128xf8E5M2>>, %arg2: !tt.ptr<i32>) -> tensor<128x128xf32, #blocked> {
  %0 = tt.load %arg2 : !tt.ptr<i32>
  %1 = tt.descriptor_load %arg1[%0, %0] : !tt.tensordesc<tensor<128x128xf8E5M2>> -> tensor<128x128xf8E5M2, #blocked1>
  %2 = ttg.convert_layout %1 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
  tt.return %3 : tensor<128x128xf32, #blocked>
}
}

// -----

// check for heuristic to increase kWidth when join is present
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @join_reshape_upcast_mma_kwidth(%84: tensor<16x256x!tt.ptr<bf16>, #blocked3>, %112: tensor<64x128x!tt.ptr<i8>, #blocked2>) -> tensor<16x64xf32, #blocked> {
      %90 = tt.load %84 : tensor<16x256x!tt.ptr<bf16>, #blocked3>
      %118 = tt.load %112, : tensor<64x128x!tt.ptr<i8>, #blocked2>
      %121:2 = tt.elementwise_inline_asm "" {constraints = "=r,=r,=r,=r,r", packed_element = 4 : i32, pure = true} %118 : tensor<64x128xi8, #blocked2> -> tensor<64x128xbf16, #blocked2>, tensor<64x128xbf16, #blocked2>
      %122 = tt.join %121#0, %121#1 : tensor<64x128xbf16, #blocked2> -> tensor<64x128x2xbf16, #blocked4>
      %123 = tt.reshape %122 : tensor<64x128x2xbf16, #blocked4> -> tensor<64x256xbf16, #blocked5>
      %124 = tt.trans %123 {order = array<i32: 1, 0>} : tensor<64x256xbf16, #blocked5> -> tensor<256x64xbf16, #blocked6>
      %125 = ttg.convert_layout %90 : tensor<16x256xbf16, #blocked3> -> tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %126 = ttg.convert_layout %124 : tensor<256x64xbf16, #blocked6> -> tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      // CHECK: {{.*}} = tt.dot {{.*}} tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = {{.*}}, kWidth = 8}>> * tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = {{.*}}, kWidth = 8}>>
      %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
      %127 = tt.dot %125, %126, %cst, inputPrecision = tf32 : tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x64xf32, #blocked>
      tt.return %127 : tensor<16x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM1:.+]] = #ttng.tensor_memory_scales_encoding
  // CHECK{LITERALE}-DAG: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
  // CHECK-LABEL: mmav5_block_scaled_8_warps
  //       CHECK:   ttng.tmem_alloc %{{.*}} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   ttng.tmem_alloc %{{.*}} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   ttng.tc_gen5_mma_scaled
  tt.func public @mmav5_block_scaled_8_warps(%a: tensor<128x256xi8, #blocked2>, %scale_a: tensor<128x8xi8, #blocked1>, %b: tensor<256x128xi8, #blocked>, %scale_b: tensor<128x8xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x256xi8, #blocked2>, tensor<128x8xi8, #blocked1> * tensor<256x128xi8, #blocked>, tensor<128x8xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}
