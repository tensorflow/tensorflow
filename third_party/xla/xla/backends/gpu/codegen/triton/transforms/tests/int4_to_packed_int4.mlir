// RUN: xla-opt --int4-to-packed-int4-rewrite %s --mlir-print-ir-after-all

module {
  tt.func @gemm_fusion_dot_2_impl(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %0 = tt.get_program_id x : i32
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %3 = arith.subi %c1_i32, %2 : i32
    %4 = arith.cmpi slt, %3, %c8_i32 : i32
    %5 = arith.select %4, %3, %c8_i32 : i32
    %6 = arith.remsi %0, %5 : i32
    %7 = arith.addi %2, %6 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %8 = arith.remsi %0, %c16_i32_0 : i32
    %9 = arith.divsi %8, %5 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.muli %7, %c128_i32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %11 = arith.addi %10, %c0_i32 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i32_1 = arith.constant 0 : i32
    %c128_i64_2 = arith.constant 128 : i64
    %c0_i32_3 = arith.constant 0 : i32
    %c128_i64_4 = arith.constant 128 : i64
    %c0_i32_5 = arith.constant 0 : i32
    %12 = arith.addi %c0_i32_3, %c0_i32_5 : i32
    %c64_i64 = arith.constant 64 : i64
    %c0_i32_6 = arith.constant 0 : i32
    %c64_i64_7 = arith.constant 64 : i64
    %c8192_i32 = arith.constant 8192 : i32
    %13 = tt.get_program_id y : i32
    %c0_i32_8 = arith.constant 0 : i32
    %14 = arith.addi %c0_i32_8, %13 : i32
    %15 = arith.muli %14, %c8192_i32 : i32
    %16 = tt.addptr %arg0, %15 : !tt.ptr<i4>, i32
    %17 = tt.make_tensor_ptr %16, [%c128_i64_2, %c64_i64_7], [%c1_i64, %c128_i64_4], [%c0_i32_1, %c0_i32_6] {order = array<i32: 1, 0>} : <tensor<128x32xi4>>
    %18 = tt.advance %17, [%10, %c0_i32_3] : <tensor<128x32xi4>>
    %c0_i32_9 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32_10 = arith.constant 0 : i32
    %19 = arith.addi %c0_i32_9, %c0_i32_10 : i32
    %c64_i64_11 = arith.constant 64 : i64
    %c0_i32_12 = arith.constant 0 : i32
    %c64_i64_13 = arith.constant 64 : i64
    %c128_i32_14 = arith.constant 128 : i32
    %20 = arith.muli %9, %c128_i32_14 : i32
    %c1_i64_15 = arith.constant 1 : i64
    %c0_i32_16 = arith.constant 0 : i32
    %21 = arith.addi %20, %c0_i32_16 : i32
    %c256_i64_17 = arith.constant 256 : i64
    %c0_i32_18 = arith.constant 0 : i32
    %c256_i64_19 = arith.constant 256 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %22 = tt.get_program_id y : i32
    %c0_i32_20 = arith.constant 0 : i32
    %23 = arith.addi %c0_i32_20, %22 : i32
    %24 = arith.muli %23, %c16384_i32 : i32
    %25 = tt.addptr %arg1, %24 : !tt.ptr<f32>, i32
    %26 = tt.make_tensor_ptr %25, [%c64_i64_13, %c256_i64_19], [%c256_i64, %c1_i64_15], [%c0_i32_12, %c0_i32_18] {order = array<i32: 1, 0>} : <tensor<32x128xf32>>
    %27 = tt.advance %26, [%c0_i32_9, %20] : <tensor<32x128xf32>>
    %c0_i32_21 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %28:3 = scf.for %arg3 = %c0_i32_21 to %c64_i32 step %c32_i32 iter_args(%arg4 = %18, %arg5 = %27, %arg6 = %cst) -> (!tt.ptr<tensor<128x32xi4>>, !tt.ptr<tensor<32x128xf32>>, tensor<128x128xf32>)  : i32 {
      %39 = tt.load %arg4 : !tt.ptr<tensor<128x32xi4>>
      %c0_i32_35 = arith.constant 0 : i32
      %c32_i32_36 = arith.constant 32 : i32
      %40 = tt.advance %arg4, [%c0_i32_35, %c32_i32_36] : <tensor<128x32xi4>>
      %41 = tt.load %arg5 : !tt.ptr<tensor<32x128xf32>>
      %c32_i32_37 = arith.constant 32 : i32
      %c0_i32_38 = arith.constant 0 : i32
      %42 = tt.advance %arg5, [%c32_i32_37, %c0_i32_38] : <tensor<32x128xf32>>
      %43 = arith.extsi %39 : tensor<128x32xi4> to tensor<128x32xi8>
      %44 = arith.sitofp %43 : tensor<128x32xi8> to tensor<128x32xf32>
      %45 = tt.dot %44, %41, %arg6 : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>
      scf.yield %40, %42, %45 : !tt.ptr<tensor<128x32xi4>>, !tt.ptr<tensor<32x128xf32>>, tensor<128x128xf32>
    }
    %c128_i32_22 = arith.constant 128 : i32
    %29 = arith.muli %7, %c128_i32_22 : i32
    %c256_i64_23 = arith.constant 256 : i64
    %c0_i32_24 = arith.constant 0 : i32
    %30 = arith.addi %29, %c0_i32_24 : i32
    %c128_i64_25 = arith.constant 128 : i64
    %c0_i32_26 = arith.constant 0 : i32
    %c128_i64_27 = arith.constant 128 : i64
    %c128_i32_28 = arith.constant 128 : i32
    %31 = arith.muli %9, %c128_i32_28 : i32
    %c1_i64_29 = arith.constant 1 : i64
    %c0_i32_30 = arith.constant 0 : i32
    %32 = arith.addi %31, %c0_i32_30 : i32
    %c256_i64_31 = arith.constant 256 : i64
    %c0_i32_32 = arith.constant 0 : i32
    %c256_i64_33 = arith.constant 256 : i64
    %c32768_i32 = arith.constant 32768 : i32
    %33 = tt.get_program_id y : i32
    %c0_i32_34 = arith.constant 0 : i32
    %34 = arith.addi %c0_i32_34, %33 : i32
    %35 = arith.muli %34, %c32768_i32 : i32
    %36 = tt.addptr %arg2, %35 : !tt.ptr<f32>, i32
    %37 = tt.make_tensor_ptr %36, [%c128_i64_27, %c256_i64_33], [%c256_i64_23, %c1_i64_29], [%c0_i32_26, %c0_i32_32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
    %38 = tt.advance %37, [%29, %31] : <tensor<128x128xf32>>
    tt.store %38, %28#2 : !tt.ptr<tensor<128x128xf32>>
    tt.return
  }
}
