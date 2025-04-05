// RUN: triton-opt %s -convert-triton-to-tritongpu=target=cuda:80 -tritongpu-remove-layout-conversions -tritongpu-pipeline=num-stages=3 -canonicalize -test-print-allocation 2>&1 | FileCheck %s

// CHECK: offset = 0, size = 32768
// CHECK: offset = 32768, size = 32768
// CHECK: size = 65536
module {
tt.func @matmul_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32_i32_i32_i32_i32_i32_i32__12c64_13c64_14c64_15c8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32) {
    %cst = arith.constant dense<true> : tensor<64x64xi1>
    %c64 = arith.constant 64 : i32
    %c0 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.cmpi slt, %8, %c8_i32 : i32
    %10 = arith.select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %17 = tt.splat %15 : i32 -> tensor<64xi32>
    %18 = arith.addi %17, %16 : tensor<64xi32>
    %19 = arith.muli %14, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %21 = tt.splat %19 : i32 -> tensor<64xi32>
    %22 = arith.addi %21, %20 : tensor<64xi32>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %24 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %25 = tt.splat %arg6 : i32 -> tensor<64x1xi32>
    %26 = arith.muli %24, %25 : tensor<64x1xi32>
    %27 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %28 = tt.splat %arg7 : i32 -> tensor<1x64xi32>
    %29 = arith.muli %27, %28 : tensor<1x64xi32>
    %30 = tt.broadcast %26 : tensor<64x1xi32> -> tensor<64x64xi32>
    %31 = tt.broadcast %29 : tensor<1x64xi32> -> tensor<64x64xi32>
    %32 = arith.addi %30, %31 : tensor<64x64xi32>
    %33 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %34 = tt.addptr %33, %32 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %35 = tt.expand_dims %23 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %36 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %37 = arith.muli %35, %36 : tensor<64x1xi32>
    %38 = tt.expand_dims %22 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %39 = tt.splat %arg9 : i32 -> tensor<1x64xi32>
    %40 = arith.muli %38, %39 : tensor<1x64xi32>
    %41 = tt.broadcast %37 : tensor<64x1xi32> -> tensor<64x64xi32>
    %42 = tt.broadcast %40 : tensor<1x64xi32> -> tensor<64x64xi32>
    %43 = arith.addi %41, %42 : tensor<64x64xi32>
    %44 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %45 = tt.addptr %44, %43 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %47:3 = scf.for %arg12 = %c0 to %arg5 step %c64 iter_args(%arg13 = %cst_0, %arg14 = %34, %arg15 = %45) -> (tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>>) : i32 {
      %76 = tt.load %arg14, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>>
      %77 = tt.load %arg15, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>>
      %78 = tt.dot %76, %77, %cst_0 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
      %79 = arith.addf %arg13, %78 : tensor<64x64xf32>
      %80 = arith.muli %arg7, %c64_i32 : i32
      %81 = tt.splat %80 : i32 -> tensor<64x64xi32>
      %82 = tt.addptr %arg14, %81 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
      %83 = arith.muli %arg8, %c64_i32 : i32
      %84 = tt.splat %83 : i32 -> tensor<64x64xi32>
      %85 = tt.addptr %arg15, %84 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
      scf.yield %79, %82, %85 : tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>>
    }
    %48 = arith.muli %12, %c64_i32 : i32
    %49 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %50 = tt.splat %48 : i32 -> tensor<64xi32>
    %51 = arith.addi %50, %49 : tensor<64xi32>
    %52 = arith.muli %14, %c64_i32 : i32
    %53 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %54 = tt.splat %52 : i32 -> tensor<64xi32>
    %55 = arith.addi %54, %53 : tensor<64xi32>
    %56 = tt.expand_dims %51 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %57 = tt.splat %arg10 : i32 -> tensor<64x1xi32>
    %58 = arith.muli %57, %56 : tensor<64x1xi32>
    %59 = tt.expand_dims %55 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %60 = tt.splat %arg11 : i32 -> tensor<1x64xi32>
    %61 = arith.muli %59, %60 : tensor<1x64xi32>
    %62 = tt.broadcast %58 : tensor<64x1xi32> -> tensor<64x64xi32>
    %63 = tt.broadcast %61 : tensor<1x64xi32> -> tensor<64x64xi32>
    %64 = arith.addi %62, %63 : tensor<64x64xi32>
    %65 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %66 = tt.addptr %65, %64 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %67 = tt.expand_dims %51 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %68 = tt.splat %arg3 : i32 -> tensor<64x1xi32>
    %69 = arith.cmpi slt, %67, %68 : tensor<64x1xi32>
    %70 = tt.expand_dims %55 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %71 = tt.splat %arg4 : i32 -> tensor<1x64xi32>
    %72 = arith.cmpi slt, %70, %71 : tensor<1x64xi32>
    %73 = tt.broadcast %69 : tensor<64x1xi1> -> tensor<64x64xi1>
    %74 = tt.broadcast %72 : tensor<1x64xi1> -> tensor<64x64xi1>
    %75 = arith.andi %73, %74 : tensor<64x64xi1>
    tt.store %66, %47#0, %75 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}
