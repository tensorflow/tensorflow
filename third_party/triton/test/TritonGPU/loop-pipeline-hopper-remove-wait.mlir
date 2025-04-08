// RUN: triton-opt %s -canonicalize -tritongpu-pipeline -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: two_dependent_dot
  tt.func public @two_dependent_dot(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = arith.divsi %2, %arg8 : i32
    %4 = arith.extsi %arg21 : i32 to i64
    %5 = arith.extsi %arg11 : i32 to i64
    %6 = arith.extsi %c0_i32 : i32 to i64
    %7 = arith.extsi %3 : i32 to i64
    %8 = arith.extsi %arg14 : i32 to i64
    %9 = arith.extsi %3 : i32 to i64
    %10 = arith.extsi %c0_i32 : i32 to i64
    %11 = arith.muli %0, %c128_i32 : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %15 = tt.splat %11 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.splat %11 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %17 = tt.splat %11 : i32 -> tensor<128xi32, #blocked1>
    %18 = arith.addi %15, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = arith.addi %16, %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %20 = arith.addi %17, %14 : tensor<128xi32, #blocked1>
    %21 = arith.mulf %arg3, %cst_4 : f32
    %22 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %23 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %24 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %25 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %26 = arith.muli %23, %25 : tensor<128x1xi32, #blocked>
    %27 = tt.splat %22 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %28 = tt.addptr %27, %26 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %31 = tt.broadcast %28 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %32 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %33 = tt.addptr %31, %32 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %34 = tt.load %33 : tensor<128x128x!tt.ptr<f16>, #blocked>
    %35 = tt.splat %21 : f32 -> tensor<128x128xf32, #blocked>
    %36 = arith.extf %34 : tensor<128x128xf16, #blocked> to tensor<128x128xf32, #blocked>
    %37 = arith.mulf %36, %35 : tensor<128x128xf32, #blocked>
    %38 = arith.truncf %37 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %39 = arith.addi %0, %c1_i32 : i32
    %40 = arith.muli %39, %c128_i32 : i32
    %41:7 = scf.for %arg22 = %c0_i32 to %40 step %c64_i32 iter_args(%arg23 = %cst_3, %arg24 = %cst_2, %arg25 = %cst_1, %arg26 = %6, %arg27 = %7, %arg28 = %9, %arg29 = %10) -> (tensor<128x128xf32, #mma1>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i64, i64, i64, i64)  : i32 {
      %69 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
      %70 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %71 = arith.extsi %70 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %72 = tt.splat %arg26 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %73 = arith.addi %71, %72 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %74 = tt.expand_dims %73 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi64, #blocked2>
      %75 = tt.broadcast %74 : tensor<128x1xi64, #blocked2> -> tensor<128x64xi64, #blocked2>
      %76 = tt.splat %c1_i64 : i64 -> tensor<128x64xi64, #blocked2>
      %77 = arith.muli %75, %76 : tensor<128x64xi64, #blocked2>
      %78 = tt.broadcast %77 : tensor<128x64xi64, #blocked2> -> tensor<128x64xi64, #blocked2>
      %79 = tt.addptr %69, %78 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi64, #blocked2>
      %80 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %81 = arith.extsi %80 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %82 = tt.splat %arg27 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %83 = arith.addi %81, %82 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi64, #blocked2>
      %85 = tt.broadcast %84 : tensor<1x64xi64, #blocked2> -> tensor<128x64xi64, #blocked2>
      %86 = tt.splat %5 : i64 -> tensor<128x64xi64, #blocked2>
      %87 = arith.muli %85, %86 : tensor<128x64xi64, #blocked2>
      %88 = tt.broadcast %87 : tensor<128x64xi64, #blocked2> -> tensor<128x64xi64, #blocked2>
      %89 = tt.addptr %79, %88 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi64, #blocked2>
      %90 = tt.load %89 : tensor<128x64x!tt.ptr<f16>, #blocked2>
      %91 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %92 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %93 = arith.extsi %92 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %94 = tt.splat %arg28 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %95 = arith.addi %93, %94 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %96 = tt.expand_dims %95 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
      %97 = tt.broadcast %96 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %98 = tt.splat %8 : i64 -> tensor<64x128xi64, #blocked>
      %99 = arith.muli %97, %98 : tensor<64x128xi64, #blocked>
      %100 = tt.broadcast %99 : tensor<64x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %101 = tt.addptr %91, %100 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %102 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = arith.extsi %102 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %104 = tt.splat %arg29 : i64 -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %105 = arith.addi %103, %104 : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %106 = tt.expand_dims %105 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked>
      %107 = tt.broadcast %106 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %108 = tt.splat %c1_i64 : i64 -> tensor<64x128xi64, #blocked>
      %109 = arith.muli %107, %108 : tensor<64x128xi64, #blocked>
      %110 = tt.broadcast %109 : tensor<64x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %111 = tt.addptr %101, %110 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %112 = tt.load %111 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %113 = ttg.local_alloc %38 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %114 = ttg.local_alloc %90 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
      %115 = ttng.warp_group_dot %113, %114, %cst :!ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      %116 = arith.truncf %115 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %117 = ttg.local_alloc %112 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %118 = ttg.convert_layout %116 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      // The first dot gets converted to dot-async + wait.  The second one
      // doesn't have a wait because the first wait is sufficient.
      // CHECK: ttng.warp_group_dot
      // CHECK: ttng.warp_group_dot_wait {{.*}}, {{.*}} {pendings = 0 : i32}
      // CHECK: ttng.warp_group_dot
      // CHECK-NOT: ttng.warp_group_dot_wait
      // CHECK: scf.yield
      %119 = ttng.warp_group_dot %118, %117, %arg23 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma1>
      %120 = arith.mulf %arg24, %arg25 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %121 = arith.addf %120, %arg25 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %122 = arith.extsi %c0_i32 : i32 to i64
      %123 = arith.addi %arg26, %122 : i64
      %124 = arith.extsi %c64_i32 : i32 to i64
      %125 = arith.addi %arg27, %124 : i64
      %126 = arith.extsi %c64_i32 : i32 to i64
      %127 = arith.addi %arg28, %126 : i64
      %128 = arith.extsi %c0_i32 : i32 to i64
      %129 = arith.addi %arg29, %128 : i64
      scf.yield %119, %121, %arg25, %123, %125, %127, %129 : tensor<128x128xf32, #mma1>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i64, i64, i64, i64
    }
    %42 = arith.addi %3, %11 : i32
    %43 = arith.extsi %arg17 : i32 to i64
    %44 = arith.extsi %42 : i32 to i64
    %45 = arith.extsi %c0_i32 : i32 to i64
    %46 = arith.truncf %41#0 : tensor<128x128xf32, #mma1> to tensor<128x128xf16, #mma1>
    %47 = ttg.convert_layout %46 : tensor<128x128xf16, #mma1> -> tensor<128x128xf16, #blocked>
    %48 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %50 = arith.extsi %49 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %51 = tt.splat %44 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = arith.addi %50, %51 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %53 = tt.expand_dims %52 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi64, #blocked>
    %54 = tt.broadcast %53 : tensor<128x1xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %55 = tt.splat %43 : i64 -> tensor<128x128xi64, #blocked>
    %56 = arith.muli %54, %55 : tensor<128x128xi64, #blocked>
    %57 = tt.broadcast %56 : tensor<128x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %58 = tt.addptr %48, %57 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    %59 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %60 = arith.extsi %59 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %61 = tt.splat %45 : i64 -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %62 = arith.addi %60, %61 : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %63 = tt.expand_dims %62 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked>
    %64 = tt.broadcast %63 : tensor<1x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %65 = tt.splat %c1_i64 : i64 -> tensor<128x128xi64, #blocked>
    %66 = arith.muli %64, %65 : tensor<128x128xi64, #blocked>
    %67 = tt.broadcast %66 : tensor<128x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %68 = tt.addptr %58, %67 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    tt.store %68, %47 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
