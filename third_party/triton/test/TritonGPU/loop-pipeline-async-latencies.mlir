// RUN: triton-opt %s --tritongpu-pipeline="num-stages=3" -canonicalize -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: matmul_kernel_tma_persistent
tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
  %0 = arith.subi %arg3, %c2_i32 : i32

  %1 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x64xf16, #shared>>
  %2 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<256x64xf16, #shared>>

  // CHECK: [[LHS_BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16,
  // CHECK: [[RHS_BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<4x256x64xf16,

  // CHECK: [[LHS_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: [[LHS_BAR0:%.*]] = ttg.memdesc_subview [[LHS_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[LHS_BAR0]]
  // CHECK-NEXT: [[LHS_BAR1:%.*]] = ttg.memdesc_subview [[LHS_BARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[LHS_BAR1]]

  // CHECK: [[RHS_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<4xi64,
  // CHECK-NEXT: [[RHS_BAR0:%.*]] = ttg.memdesc_subview [[RHS_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[RHS_BAR0]]
  // CHECK-NEXT: [[RHS_BAR1:%.*]] = ttg.memdesc_subview [[RHS_BARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[RHS_BAR1]]
  // CHECK-NEXT: [[RHS_BAR2:%.*]] = ttg.memdesc_subview [[RHS_BARS]][%c2_i32]
  // CHECK-NEXT: ttng.init_barrier [[RHS_BAR2]]
  // CHECK-NEXT: [[RHS_BAR3:%.*]] = ttg.memdesc_subview [[RHS_BARS]][%c3_i32]
  // CHECK-NEXT: ttng.init_barrier [[RHS_BAR3]]

  // CHECK: [[MASK0:%.*]] = arith.cmpi sgt, %arg3, %c0_i32
  // CHECK-NEXT: ttng.barrier_expect [[RHS_BAR0]], 32768, [[MASK0]]
  // CHECK-NEXT: [[RHS_BUF0:%.*]] = ttg.memdesc_subview [[RHS_BUFFERS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg1[%c0_i32, %c0_i32] [[RHS_BUF0]], [[RHS_BAR0]], [[MASK0]]

  // CHECK: [[MASK1:%.*]] = arith.cmpi sgt, %arg3, %c1_i32
  // CHECK-NEXT: ttng.barrier_expect [[RHS_BAR1]], 32768, [[MASK1]]
  // CHECK-NEXT: [[RHS_BUF1:%.*]] = ttg.memdesc_subview [[RHS_BUFFERS]][%c1_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg1[%c0_i32, %c1_i32] [[RHS_BUF1]], [[RHS_BAR1]], [[MASK1]]

  // CHECK: [[MASK2:%.*]] = arith.cmpi sgt, %arg3, %c2_i32

  // CHECK-NEXT: ttng.barrier_expect [[LHS_BAR0]], 16384, [[MASK0]]
  // CHECK-NEXT: [[LHS_BUF0:%.*]] = ttg.memdesc_subview [[LHS_BUFFERS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] [[LHS_BUF0]], [[LHS_BAR0]], [[MASK0]]

  // CHECK: ttng.barrier_expect [[RHS_BAR2]], 32768, [[MASK2]]
  // CHECK-NEXT: [[RHS_BUF2:%.*]] = ttg.memdesc_subview [[RHS_BUFFERS]][%c2_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg1[%c0_i32, %c2_i32] [[RHS_BUF2]], [[RHS_BAR2]], [[MASK2]]

  %true = arith.constant true
  %false = arith.constant false

  // CHECK: scf.for [[I:%.*]] = %c0_i32 to
  // CHECK-SAME: iter_args([[ACCUM:%arg[0-9]+]] = %cst

  // CHECK-SAME: [[NEXT_LHS_BUF_IDX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LHS_BUF_IDX:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME: [[LHS_PHASE_ARG:%arg[0-9]+]] = %c0_i32

  // CHECK-SAME: [[NEXT_RHS_BUF_IDX:%arg[0-9]+]] = %c2_i32
  // CHECK-SAME: [[RHS_BUF_IDX:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME: [[RHS_PHASE_ARG:%arg[0-9]+]] = %c0_i32
  %3 = scf.for %arg6 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<128x256xf32, #mma>)  : i32 {
    // CHECK: [[RHS_MAX_ITER:%.*]] = arith.subi %arg3, %c3_i32
    // CHECK-NEXT: [[RHS_MASK:%.*]] = arith.cmpi slt, [[I]], [[RHS_MAX_ITER]]
    // CHECK: [[LHS_MAX_ITER:%.*]] = arith.subi %arg3, %c1_i32
    // CHECK-NEXT: [[LHS_MASK:%.*]] = arith.cmpi slt, [[I]], [[LHS_MAX_ITER]]

    // Compute RHS buffer index modulo 4.
    // CHECK: [[V0:%.*]] = arith.addi [[RHS_BUF_IDX]], %c1_i32
    // CHECK-NEXT: [[V1:%.*]] = arith.cmpi slt, [[V0]], %c4_i32
    // CHECK-NEXT: [[RHS_BUF_IDX:%.*]] = arith.select [[V1]], [[V0]], %c0_i32

    // Compute RHS phase index modulo 4.
    // CHECK: [[V0:%.*]] = arith.xori [[RHS_PHASE_ARG]], %c1_i32
    // CHECK-NEXT: [[RHS_PHASE:%.*]] = arith.select [[V1]], [[RHS_PHASE_ARG]], [[V0]]

    // Compute LHS buffer index modulo 2.
    // CHECK: [[V0:%.*]] = arith.addi [[LHS_BUF_IDX]], %c1_i32
    // CHECK-NEXT: [[V1:%.*]] = arith.cmpi slt, [[V0]], %c2_i32
    // CHECK-NEXT: [[LHS_BUF_IDX:%.*]] = arith.select [[V1]], [[V0]], %c0_i32

    // Compute LHS phase index modulo 2.
    // CHECK: [[V0:%.*]] = arith.xori [[LHS_PHASE_ARG]], %c1_i32
    // CHECK-NEXT: [[LHS_PHASE:%.*]] = arith.select [[V1]], [[LHS_PHASE_ARG]], [[V0]]

    // CHECK: [[LHS_MBAR:%.*]] = ttg.memdesc_subview [[LHS_BARS]][[[LHS_BUF_IDX]]]
    // CHECK-NEXT: ttng.wait_barrier [[LHS_MBAR]], [[LHS_PHASE]]

    // CHECK: [[RHS_MBAR:%.*]] = ttg.memdesc_subview [[RHS_BARS]][[[RHS_BUF_IDX]]]
    // CHECK-NEXT: ttng.wait_barrier [[RHS_MBAR]], [[RHS_PHASE]]

    %4 = tt.descriptor_load %1[%c0_i32, %arg6] {tt_latency = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    %5 = ttg.local_alloc %4 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %6 = tt.descriptor_load %2[%c0_i32, %arg6] {tt_latency = 3 : i32} : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked>
    %7 = ttg.local_alloc %6 : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
    %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
    %9 = ttng.warp_group_dot %5, %8, %arg7 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<128x256xf32, #mma>

    // CHECK: [[V0:%.*]] = arith.addi [[NEXT_LHS_BUF_IDX]], %c1_i32
    // CHECK-NEXT: [[V1:%.*]] = arith.cmpi slt, [[V0]], %c2_i32
    // CHECK-NEXT: [[NEXT_LHS_BUF_IDX:%.*]] = arith.select [[V1]], [[V0]], %c0_i32
    // CHECK-NEXT: [[NEXT_LHS_BAR:%.*]] = ttg.memdesc_subview [[LHS_BARS]][[[NEXT_LHS_BUF_IDX]]]
    // CHECK-NEXT: ttng.barrier_expect [[NEXT_LHS_BAR]], 16384, [[LHS_MASK]]

    // CHECK-NEXT: [[NEXT_LHS_BUF:%.*]] = ttg.memdesc_subview [[LHS_BUFFERS]][[[NEXT_LHS_BUF_IDX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_LHS_IDX:%.*]] = arith.addi [[I]], %c1_i32
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg0[%c0_i32, [[NEXT_LHS_IDX]]] [[NEXT_LHS_BUF]], [[NEXT_LHS_BAR]], [[LHS_MASK]]

    // CHECK: [[V0:%.*]] = arith.addi [[NEXT_RHS_BUF_IDX]], %c1_i32
    // CHECK-NEXT: [[V1:%.*]] = arith.cmpi slt, [[V0]], %c4_i32
    // CHECK-NEXT: [[NEXT_RHS_BUF_IDX:%.*]] = arith.select [[V1]], [[V0]], %c0_i32
    // CHECK-NEXT: [[NEXT_RHS_BAR:%.*]] = ttg.memdesc_subview [[RHS_BARS]][[[NEXT_RHS_BUF_IDX]]]
    // CHECK-NEXT: ttng.barrier_expect [[NEXT_RHS_BAR]], 32768, [[RHS_MASK]]

    // CHECK-NEXT: [[NEXT_RHS_BUF:%.*]] = ttg.memdesc_subview [[RHS_BUFFERS]][[[NEXT_RHS_BUF_IDX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_RHS_IDX:%.*]] = arith.addi [[I]], %c3_i32
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local %arg1[%c0_i32, [[NEXT_RHS_IDX]]] [[NEXT_RHS_BUF]], [[NEXT_RHS_BAR]], [[RHS_MASK]]

    %10 = arith.cmpi eq, %arg3, %0 : i32
    scf.if %10 {
      %11 = arith.truncf %9 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %12 = ttg.convert_layout %11 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %13 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x256xf16, #shared>>
      tt.descriptor_store %13[%c0_i32, %c0_i32], %12 : !tt.tensordesc<tensor<128x256xf16, #shared>>, tensor<128x256xf16, #blocked1>
    }
    // CHECK: yield %{{.*}}, [[NEXT_LHS_BUF_IDX]], [[LHS_BUF_IDX]], [[LHS_PHASE]], [[NEXT_RHS_BUF_IDX]], [[RHS_BUF_IDX]], [[RHS_PHASE]]
    scf.yield %9 : tensor<128x256xf32, #mma>
  } {tt.num_stages = 4 : i32}
  tt.return
}

}
