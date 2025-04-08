// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline=num_stages=2 -canonicalize | FileCheck %s --check-prefixes=COMMON,SYNC
// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=2 use_async_copy=1" -canonicalize | FileCheck %s --check-prefixes=COMMON,ASYNC

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: tt.func @load_two_users
  tt.func @load_two_users(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // COMMON: ttg.local_store
    // COMMON: scf.for
    // COMMON:   tt.load
    // COMMON:   tt.dot
    // COMMON:   tt.dot
    // COMMON:   ttg.local_store
    // COMMON:   scf.yield
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = ttg.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = ttg.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
      %25 = ttg.memdesc_trans %24 {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared1, #smem, mutable>
      %26 = ttg.local_load %25 : !ttg.memdesc<16x64xf16, #shared1, #smem, mutable> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// COMMON-LABEL: tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de
// COMMON-NOT:  ttg.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 : !tt.ptr<i64>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addi %5, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %3 : i64 -> tensor<64x1xi64, #blocked>
    %9 = arith.extsi %7 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %10 = arith.addi %8, %9 : tensor<64x1xi64, #blocked>
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = tt.splat %11 : i64 -> tensor<64x1xi64, #blocked>
    %13 = arith.muli %10, %12 : tensor<64x1xi64, #blocked>
    %14 = arith.muli %2, %arg5 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = tt.splat %15 : i64 -> tensor<64x1xi64, #blocked>
    %17 = arith.addi %13, %16 : tensor<64x1xi64, #blocked>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %24 = arith.muli %20, %22 : tensor<1x64xi32, #blocked>
    %25 = arith.muli %21, %23 : tensor<1x64xi32, #blocked1>
    %26 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %27 = arith.extsi %24 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %28 = arith.extsi %25 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %30 = arith.addi %26, %29 : tensor<64x64xi64, #blocked>
    %31 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %33 = tt.splat %3 : i64 -> tensor<32x1xi64, #blocked1>
    %34 = arith.extsi %32 : tensor<32x1xi32, #blocked1> to tensor<32x1xi64, #blocked1>
    %35 = arith.addi %33, %34 : tensor<32x1xi64, #blocked1>
    %36 = tt.splat %11 : i64 -> tensor<32x1xi64, #blocked1>
    %37 = arith.muli %35, %36 : tensor<32x1xi64, #blocked1>
    %38 = tt.splat %15 : i64 -> tensor<32x1xi64, #blocked1>
    %39 = arith.addi %37, %38 : tensor<32x1xi64, #blocked1>
    %40 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %41 = tt.broadcast %28 : tensor<1x64xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %42 = arith.addi %40, %41 : tensor<32x64xi64, #blocked1>
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %47 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %48 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %49 = arith.muli %45, %47 : tensor<1x32xi32, #blocked1>
    %50 = arith.muli %46, %48 : tensor<1x32xi32, #blocked>
    %51 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %52 = arith.extsi %49 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
    %53 = arith.extsi %50 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %54 = tt.broadcast %52 : tensor<1x32xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %55 = arith.addi %51, %54 : tensor<32x32xi64, #blocked1>
    %56 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %57 = tt.addptr %56, %30 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi64, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %59 = tt.addptr %58, %42 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi64, #blocked1>
    %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi64, #blocked1>
    %62 = tt.load %57 : tensor<64x64x!tt.ptr<f32>, #blocked>
    %63 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xf32, #mma>)  : i32 {
      %70 = tt.load %59 : tensor<32x64x!tt.ptr<f32>, #blocked1>
      %71 = ttg.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = ttg.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
      %73 = ttg.memdesc_trans %72 {order=array<i32: 1,0>} : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable>
      %74 = ttg.local_load %73 : !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %77 = ttg.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = ttg.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = ttg.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
} // end module

// -----

// Disable pipelining for loops that contain barrier.
//   Barriers are problematic since they are not chained to any other operation.
// COMMON-LABEL: tt.func public @add_barrier_kernel
// COMMON:  scf.for
// COMMON:    tt.load
// COMMON:    gpu.barrier
// COMMON:    tt.store
// COMMON-NOT:  gpu.barrier
// COMMON:  tt.return

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @add_barrier_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %cval_f32 = arith.constant dense<0.3> : tensor<1024xf32, #blocked>
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %arg3 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
      gpu.barrier
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %15 = arith.addf %12, %cval_f32 : tensor<1024xf32, #blocked>
      tt.store %16, %15 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
} // end module

// -----

// COMMON-NOT: #ttg.swizzled_shared<{{.*}} order = [2, 0, 1]
// COMMON: #ttg.swizzled_shared<{{.*}} order = [2, 1, 0]
// COMMON-NOT: #ttg.swizzled_shared<{{.*}} order = [2, 0, 1]

// COMMON-LABEL: tt.func public @slowest_dim_is_batch
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [4, 1, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [16, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [16, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @slowest_dim_is_batch(%arg0: tensor<1x512x!tt.ptr<f32>, #blocked2>, %arg1: tensor<64x8x32x!tt.ptr<f32>, #blocked1>, %arg2: tensor<64x1x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1x32xf32, #blocked>
    %cst_0 = arith.constant dense<512> : tensor<1x512xi32, #blocked2>
    %cst_1 = arith.constant dense<128> : tensor<64x8x32xi32, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %33:3 = scf.for %arg7 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %arg0, %arg10 = %arg1) -> (tensor<64x1x32xf32, #blocked>, tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<64x8x32x!tt.ptr<f32>, #blocked1>)  : i32 {
      %39 = tt.load %arg9 : tensor<1x512x!tt.ptr<f32>, #blocked2>
      %40 = tt.load %arg10 : tensor<64x8x32x!tt.ptr<f32>, #blocked1>
      %41 = tt.reshape %39 allow_reorder : tensor<1x512xf32, #blocked2> -> tensor<64x1x8xf32, #blocked5>
      %43 = ttg.convert_layout %41 : tensor<64x1x8xf32, #blocked5> -> tensor<64x1x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %44 = ttg.convert_layout %40 : tensor<64x8x32xf32, #blocked1> -> tensor<64x8x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %45 = tt.dot %43, %44, %arg8 : tensor<64x1x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x8x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x1x32xf32, #blocked>
      %46 = tt.addptr %arg9, %cst_0 : tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<1x512xi32, #blocked2>
      %47 = tt.addptr %arg10, %cst_1 : tensor<64x8x32x!tt.ptr<f32>, #blocked1>, tensor<64x8x32xi32, #blocked1>
      scf.yield %45, %46, %47 : tensor<64x1x32xf32, #blocked>, tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<64x8x32x!tt.ptr<f32>, #blocked1>
    }
    tt.store %arg2, %33#0 : tensor<64x1x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that the stream pipeliner updates the resulting memory layout of transpose ops to mutable if immutable local buffers are replaced
// COMMON-LABEL: loop_with_dot_and_transpose
// COMMON: ttg.local_alloc {{.*}}, mutable>
// COMMON: ttg.memdesc_trans {{.*}}, mutable> -> {{.*}}, mutable>

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_with_dot_and_transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg4: tensor<32x32x!tt.ptr<f32>, #blocked1>, %arg5: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg3 = %cst) -> (tensor<32x32xf32, #blocked>)  : i32 {
      %2 = tt.load %arg4 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %3 = ttg.local_alloc %2 : (tensor<32x32xf32, #blocked1>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf32, #shared, #smem> -> !ttg.memdesc<32x32xf32, #shared1, #smem>
      %5 = ttg.local_load %4 : !ttg.memdesc<32x32xf32, #shared1, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %6 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %7 = tt.dot %6, %5, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
      scf.yield %7 : tensor<32x32xf32, #blocked>
    }
    tt.store %arg5, %0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that the stream pipeliner updates atomic op in the k-loop correctly
// COMMON-LABEL: _triton_gemm_kernel_atomic_rmw
// COMMON:  scf.for
// COMMON: tt.atomic_rmw fadd, acq_rel, gpu
// COMMON:  tt.dot
// COMMON: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_triton_gemm_kernel_atomic_rmw(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %2 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<32x1xi32, #blocked>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<32x32xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %3 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %7 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %17 = tt.splat %arg3 : i32 -> tensor<32x1xi32, #blocked>
    %18 = arith.cmpi slt, %1, %17 : tensor<32x1xi32, #blocked>
    %19 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked>
    %20 = arith.cmpi slt, %5, %19 : tensor<1x32xi32, #blocked>
    %21 = tt.broadcast %18 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %22 = tt.broadcast %20 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %23 = arith.andi %21, %22 : tensor<32x32xi1, #blocked>
    %24 = arith.addi %arg3, %c31_i32 : i32
    %25 = arith.divsi %24, %c32_i32 : i32
    %26 = arith.muli %arg4, %c32_i32 : i32
    %27 = tt.splat %26 : i32 -> tensor<32x32xi32, #blocked>
    %28:3 = scf.for %arg5 = %c0_i32 to %25 step %c1_i32 iter_args(%arg6 = %cst_0, %arg7 = %10, %arg8 = %12) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>)  : i32 {
      %32 = tt.load %arg7 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %33 = tt.load %arg8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %34 = ttg.convert_layout %32 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %35 = ttg.convert_layout %33 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %36 = tt.dot %34, %35, %arg6 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
      %37 = tt.addptr %arg7, %cst : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %38 = tt.addptr %arg8, %27 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %39 = arith.truncf %36 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
      %40 = ttg.convert_layout %39 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked>
      %41 = tt.atomic_rmw fadd, acq_rel, gpu, %16, %40, %23 : (tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xf16, #blocked>, tensor<32x32xi1, #blocked>) -> tensor<32x32xf16, #blocked>
      scf.yield %36, %37, %38 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>
    }
    %29 = arith.truncf %28#0 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %30 = ttg.convert_layout %16 : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #mma>
    %31 = ttg.convert_layout %23 : tensor<32x32xi1, #blocked> -> tensor<32x32xi1, #mma>
    tt.store %30, %29, %31 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

// Check that we can pipeline scaled dot with linear layout
// COMMON-LABEL: mxfp8_mxfp4_matmul

// Prologue
// COMMON-COUNT-3: ttg.local_alloc
// COMMON-COUNT-3: tt.load
// COMMON-COUNT-3: ttg.local_store

// Main loop
//         COMMON: scf.for
// COMMON-COUNT-3:   ttg.local_load
//         COMMON:   tt.dot_scaled
//         COMMON:   scf.yield

// Epilogue
// COMMON-COUNT-3: ttg.local_load
//         COMMON: scf.if
//         COMMON:   tt.dot_scaled
// COMMON-COUNT-2:   scf.yield
// COMMON-COUNT-3: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 2], [0, 4], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mxfp8_mxfp4_matmul(
      %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32},
      %71: tensor<128x256x!tt.ptr<f32>, #blocked3>) {
    %cst = arith.constant dense<256> : tensor<128x256xi32, #blocked>
    %cst_0 = arith.constant dense<8> : tensor<256x8xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked2>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c255_i32 = arith.constant 255 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %8 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %10 = arith.addi %8, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = arith.addi %9, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %12 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.remsi %10, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.muli %4, %c256_i32 : i32
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %18 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %21 = arith.addi %18, %15 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = arith.addi %19, %16 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = arith.addi %20, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %24 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %25 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %26 = arith.remsi %21, %24 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.remsi %22, %25 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %28 = tt.expand_dims %26 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %29 = tt.splat %arg7 : i32 -> tensor<256x1xi32, #blocked1>
    %30 = arith.muli %28, %29 : tensor<256x1xi32, #blocked1>
    %31 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked1>
    %32 = tt.addptr %31, %30 : tensor<256x1x!tt.ptr<i8>, #blocked1>, tensor<256x1xi32, #blocked1>
    %33 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %35 = tt.broadcast %32 : tensor<256x1x!tt.ptr<i8>, #blocked1> -> tensor<256x8x!tt.ptr<i8>, #blocked1>
    %36 = tt.broadcast %34 : tensor<1x8xi32, #blocked1> -> tensor<256x8xi32, #blocked1>
    %37 = tt.addptr %35, %36 : tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<256x8xi32, #blocked1>
    %38 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %39 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %40 = arith.muli %38, %39 : tensor<128x1xi32, #blocked>
    %41 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = tt.expand_dims %41 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %43 = tt.broadcast %40 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %44 = tt.broadcast %42 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %45 = arith.addi %43, %44 : tensor<128x256xi32, #blocked>
    %46 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
    %47 = tt.addptr %46, %45 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
    %48 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %49 = tt.expand_dims %48 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %50 = tt.splat %arg9 : i32 -> tensor<128x1xi32, #blocked>
    %51 = arith.muli %49, %50 : tensor<128x1xi32, #blocked>
    %52 = tt.expand_dims %27 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %53 = tt.broadcast %51 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %54 = tt.broadcast %52 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %55 = arith.addi %53, %54 : tensor<128x256xi32, #blocked>
    %56 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>, #blocked>
    %57 = tt.addptr %56, %55 : tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<128x256xi32, #blocked>
    %58 = arith.addi %arg6, %c255_i32 : i32
    %59 = arith.divsi %58, %c256_i32 : i32
    %60 = arith.muli %arg9, %c128_i32 : i32
    %61 = tt.splat %60 : i32 -> tensor<128x256xi32, #blocked>
    %62:5 = scf.for %arg11 = %c0_i32 to %59 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %47, %arg14 = %57, %arg15 = %37, %arg16 = %cst_3)
      -> (tensor<128x256xf32, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<128x256xf32, #mma>)  : i32 {
      %80 = tt.load %arg13 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
      %81 = tt.load %arg14 : tensor<128x256x!tt.ptr<i8>, #blocked>
      %82 = tt.load %arg15 : tensor<256x8x!tt.ptr<i8>, #blocked1>
      %83 = ttg.convert_layout %80 : tensor<128x256xf8E5M2, #blocked> -> tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %84 = ttg.convert_layout %81 : tensor<128x256xi8, #blocked> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %85 = ttg.convert_layout %82 : tensor<256x8xi8, #blocked1> -> tensor<256x8xi8, #linear1>
      %86 = tt.dot_scaled %83 scale %cst_1, %84 scale %85, %arg16 lhs = e5m2 rhs = e2m1 {fastMath = false} : tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<128x256xf32, #mma>
      %87 = ttg.convert_layout %86 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked2>
      %88 = tt.addptr %arg13, %cst : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
      %89 = tt.addptr %arg14, %61 : tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<128x256xi32, #blocked>
      %90 = tt.addptr %arg15, %cst_0 : tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<256x8xi32, #blocked1>
      scf.yield %87, %88, %89, %90, %86 : tensor<128x256xf32, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32}
    %79 = ttg.convert_layout %62#0 : tensor<128x256xf32, #blocked2> -> tensor<128x256xf32, #blocked3>
    tt.store %71, %79 : tensor<128x256x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

// Check that we can pipeline a simple matmul kernel
// Note: Currently AsyncCopy is only used for the second operand because we do not support (actual) swizzled shared encodings

// COMMON-LABEL: simple_matmul_kernel

// Prologue
// COMMON-COUNT-2: ttg.local_alloc
  // SYNC-COUNT-2: tt.load
  // SYNC-COUNT-2: ttg.local_store
  //
  // ASYNC: tt.load
  // ASYNC: ttg.local_store
  // ASYNC: ttg.async_copy_global_to_local

// Main loop
//         COMMON:   scf.for
//
  // SYNC-COUNT-2:   ttg.local_load
  //         SYNC:   tt.dot
  //         SYNC:   scf.yield
  //
  //         ASYNC:    ttg.local_load
  //         ASYNC:    ttg.async_wait
  //         ASYNC:    ttg.local_load
  //         ASYNC:    ttg.dot
  //         ASYNC:    ttg.async_copy_global_to_local

// Epilogue
// COMMON-COUNT-2: ttg.local_load
//         COMMON: scf.if
//         COMMON:   tt.dot
// COMMON-COUNT-2:   scf.yield
// COMMON-COUNT-2: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @simple_matmul_kernel(%test: tensor<1x64xi32, #blocked1>, %arg0: tensor<64x64x!tt.ptr<f16>, #mma>, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x64xi32, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = arith.muli %arg1, %c64_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.addi %2, %0 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = arith.remsi %3, %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %11 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %13 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %14 = tt.addptr %13, %12 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %15:3 = scf.for %arg11 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg12 = %cst_1, %arg13 = %10, %arg14 = %14) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %17 = tt.load %arg13 : tensor<64x32x!tt.ptr<f16>, #blocked>
      %18 = tt.load %arg14 : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %19 = ttg.convert_layout %17 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %20 = ttg.convert_layout %18 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %21 = tt.dot %19, %20, %arg12, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %22 = tt.addptr %arg13, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %23 = tt.addptr %arg14, %cst_0 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %21, %22, %23 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>
    }
    %16 = arith.truncf %15#0 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    tt.store %arg0, %16 : tensor<64x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}
