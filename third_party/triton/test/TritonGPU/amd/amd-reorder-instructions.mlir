// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   ttg.local_load %[[ALLOC]]
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = ttg.local_load %10 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
// Move loads (and independent local_stores) as early as possible.
// For example in the matmul_loop below, the scf.for loop looks like this after pipeliner:
//   scf.for ... {
//     // stage 1
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     // stage 0
//     %aptr = tt.addptr %aptr, %k
//     %a_next = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next = tt.load %bptr
//     tt.local_store %a_next
//     tt.local_store %b_next
//     yield
//   }
//
//  Solution for num_stages=2 :
//   scf.for ... {
//     // stage 0.a
//     %aptr = tt.addptr %aptr, %k
//     %a_next = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next = tt.load %bptr
//     // stage 1
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     // stage 0.b
//     tt.local_store %a_next
//     tt.local_store %b_next
//     yield
//   }
//
//  Solution for num_stages=3 (double-buffered) :
//   scf.for ... {
//     // stage 1
//     tt.local_store %a_next_1
//     tt.local_store %b_next_1
//     // stage 0
//     %aptr = tt.addptr %aptr, %k
//     %a_next_2 = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next_2 = tt.load %bptr
//     // stage 2
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     yield
//   }

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared4 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {

// CHECK-LABEL:  tt.func @matmul_loop
// CHECK:  %{{.*}}:6 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})
// Stage 0.a
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]]
//        CHECK:       %[[ADDPTR_25:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[SPLAT_26:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_27:.*]] = tt.load %[[ADDPTR_25]], %[[SPLAT_26]]
// Stage 1
//        CHECK:       %[[LOCAL_LOAD_28:.*]] = ttg.local_load %[[ARG10]]
//        CHECK:       %[[LOCAL_LOAD_29:.*]] = ttg.local_load %[[ARG11]]
//        CHECK:       %[[MULF_30:.*]] = arith.mulf %[[LOCAL_LOAD_29]], %{{.*}}
//        CHECK:       %[[DOT_31:.*]] = tt.dot %[[LOCAL_LOAD_28]], %[[MULF_30]], %[[ARG8]]
// Stage 0.b
//        CHECK:       %[[ADDI_32:.*]] = arith.addi %[[ARG9]], %{{.*}}
//        CHECK:       %[[CMPI_33:.*]] = arith.cmpi slt, %[[ADDI_32]], %{{.*}}
//        CHECK:       %[[SELECT_34:.*]] = arith.select %[[CMPI_33]], %[[ADDI_32]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_35:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_35]]
//        CHECK:       %[[MEMDESC_SUBVIEW_36:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_36]]
//        CHECK:       scf.yield %[[ADDPTR_20]], %[[ADDPTR_25]], %[[DOT_31]], %[[SELECT_34]], %[[MEMDESC_SUBVIEW_35]], %[[MEMDESC_SUBVIEW_36]]
//        CHECK:   }

  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = ttg.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
    ttg.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
    %18 = ttg.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
    ttg.local_store %16, %18 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
    %19:6 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %17, %arg11 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>) {
      %20 = arith.subi %arg1, %arg2 : index
      %21 = arith.cmpi slt, %arg5, %20 : index
      %22 = ttg.local_load %arg10 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %23 = ttg.local_load %arg11 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %24 = arith.mulf %23, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %25 = tt.dot %22, %24, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %26 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %27 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %28 = tt.splat %21 : i1 -> tensor<128x32xi1, #blocked1>
      %29 = tt.load %26, %28 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %30 = tt.splat %21 : i1 -> tensor<32x128xi1, #blocked>
      %31 = tt.load %27, %30, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %32 = arith.addi %arg9, %c1_i32 : i32
      %33 = arith.cmpi slt, %32, %c1_i32 : i32
      %34 = arith.select %33, %32, %c0_i32 : i32
      %35 = ttg.memdesc_subview %10[%34, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
      ttg.local_store %29, %35 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
      %36 = ttg.memdesc_subview %11[%34, %c0_i32, %c0_i32] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
      ttg.local_store %31, %36 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
      scf.yield %26, %27, %25, %34, %35, %36 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
    }
    ttg.local_dealloc %10 : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    tt.return %19#2 : tensor<128x128xf32, #mma>
  }


// This example tests that tt.load overlaps with independent ttg.local_store which
// overlaps with independent tt.dot.
// num_stages == 3, double buffered

// CHECK-LABEL:  tt.func @matmul_loop_mb
// CHECK:  %{{.*}}:8 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 1
//        CHECK:       %[[ADDI_28:.*]] = arith.addi %[[ARG9]], %{{.*}}
//        CHECK:       %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
//        CHECK:       %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_31:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_30]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[ARG12]], %[[MEMDESC_SUBVIEW_31]]
//        CHECK:       %[[MEMDESC_SUBVIEW_32:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_30]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[ARG13]], %[[MEMDESC_SUBVIEW_32]]
// Stage 1
//        CHECK:       %[[ADDPTR_33:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//        CHECK:       %[[MULI_34:.*]] = arith.muli %{{.*}}, %{{.*}}
//        CHECK:       %[[SUBI_35:.*]] = arith.subi %{{.*}}, %[[MULI_34]]
//        CHECK:       %[[CMPI_36:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_35]]
//        CHECK:       %[[SPLAT_37:.*]] = tt.splat %[[CMPI_36]]
//        CHECK:       %[[LOAD_38:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_37]]
//        CHECK:       %[[ADDPTR_39:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[SPLAT_40:.*]] = tt.splat %[[CMPI_36]]
//        CHECK:       %[[LOAD_41:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_40]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_42:.*]] = ttg.local_load %[[ARG10]]
//        CHECK:       %[[LOCAL_LOAD_43:.*]] = ttg.local_load %[[ARG11]]
//        CHECK:       %[[MULF_44:.*]] = arith.mulf %[[LOCAL_LOAD_43]], %{{.*}}
//        CHECK:       %[[DOT_45:.*]] = tt.dot %[[LOCAL_LOAD_42]], %[[MULF_44]], %[[ARG8]]
//        CHECK:       scf.yield %[[ADDPTR_33]], %[[ADDPTR_39]], %[[DOT_45]], %[[SELECT_30]], %[[MEMDESC_SUBVIEW_31]], %[[MEMDESC_SUBVIEW_32]], %[[LOAD_38]], %[[LOAD_41]]
//        CHECK:   }

  tt.func @matmul_loop_mb(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = arith.addi %arg0, %arg2 : index
    %18 = arith.cmpi slt, %17, %arg1 : index
    %19 = tt.addptr %4, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %20 = tt.addptr %9, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %21 = tt.splat %18 : i1 -> tensor<128x32xi1, #blocked1>
    %22 = tt.load %19, %21 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %23 = tt.splat %18 : i1 -> tensor<32x128xi1, #blocked>
    %24 = tt.load %20, %23, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %25 = ttg.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
    ttg.local_store %14, %25 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
    %26 = ttg.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
    ttg.local_store %16, %26 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
    %27:8 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %19, %arg7 = %20, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %25, %arg11 = %26, %arg12 = %22, %arg13 = %24) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>) {
      %28 = arith.muli %arg2, %c2 : index
      %29 = arith.subi %arg1, %28 : index
      %30 = arith.cmpi slt, %arg5, %29 : index
      %31 = ttg.local_load %arg10 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = ttg.local_load %arg11 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %33 = arith.mulf %32, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %34 = tt.dot %31, %33, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %35 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %36 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %37 = tt.splat %30 : i1 -> tensor<128x32xi1, #blocked1>
      %38 = tt.load %35, %37 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %39 = tt.splat %30 : i1 -> tensor<32x128xi1, #blocked>
      %40 = tt.load %36, %39, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %41 = arith.addi %arg9, %c1_i32 : i32
      %42 = arith.cmpi slt, %41, %c2_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      %44 = ttg.memdesc_subview %10[%43, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
      ttg.local_store %arg12, %44 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>
      %45 = ttg.memdesc_subview %11[%43, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
      ttg.local_store %arg13, %45 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>
      scf.yield %35, %36, %34, %43, %44, %45, %38, %40 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 1x128x32>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 1x32x128>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>
    }
    ttg.local_dealloc %10 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    tt.return %27#2 : tensor<128x128xf32, #mma>
  }

// This example shows dependent loads and verifies all are moved early.
// CHECK-LABEL:  tt.func @indirect_bmm_vector
// CHECK:  %{{.*}}:7 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 0
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]]
// Stage 1.a
//        CHECK:       %[[EXPAND_DIMS_25:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
//        CHECK:       %[[BROADCAST_26:.*]] = tt.broadcast %[[EXPAND_DIMS_25]]
//        CHECK:       %[[MULI_27:.*]] = arith.muli %{{.*}}, %[[BROADCAST_26]]
//        CHECK:       %[[ADDPTR_28:.*]] = tt.addptr %{{.*}}, %[[MULI_27]]
//        CHECK:       %[[SPLAT_29:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_30:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_29]]
//        CHECK:       %[[ADDPTR_31:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//        CHECK:       %[[SUBI_32:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_33:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_32]]
//        CHECK:       %[[SPLAT_34:.*]] = tt.splat %[[CMPI_33]]
//        CHECK:       %[[LOAD_35:.*]] = tt.load %[[ADDPTR_31]], %[[SPLAT_34]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_36:.*]] = ttg.local_load %[[ARG11]]
//        CHECK:       %[[LOCAL_LOAD_37:.*]] = ttg.local_load %[[ARG12]]
//        CHECK:       %[[DOT_38:.*]] = tt.dot %[[LOCAL_LOAD_36]], %[[LOCAL_LOAD_37]], %[[ARG7]]
// Stage 1.b
//        CHECK:       %[[ADDI_39:.*]] = arith.addi %[[ARG10]], %{{.*}}
//        CHECK:       %[[CMPI_40:.*]] = arith.cmpi slt, %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[SELECT_41:.*]] = arith.select %[[CMPI_40]], %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_42:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_42]]
//        CHECK:       %[[MEMDESC_SUBVIEW_43:.*]] = ttg.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       ttg.local_store %[[LOAD_30]], %[[MEMDESC_SUBVIEW_43]]
//        CHECK:       scf.yield %[[DOT_38]], %[[ADDPTR_20]], %[[ADDPTR_31]], %[[SELECT_41]], %[[MEMDESC_SUBVIEW_42]], %[[MEMDESC_SUBVIEW_43]], %[[LOAD_35]]
//        CHECK:   }

  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %2 = arith.cmpi sgt, %arg1, %c0 : index
    %3 = tt.splat %2 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.load %arg3, %3 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = arith.cmpi sgt, %arg1, %c1 : index
    %6 = tt.addptr %arg3, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.splat %2 : i1 -> tensor<16x16xi1, #blocked1>
    %8 = tt.load %arg2, %7 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %9 = tt.expand_dims %4 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %10 = tt.broadcast %9 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %11 = arith.muli %arg0, %10 : tensor<16x16xi64, #blocked>
    %12 = tt.addptr %arg5, %11 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %13 = tt.splat %2 : i1 -> tensor<16x16xi1, #blocked>
    %14 = tt.load %12, %13 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %15 = tt.splat %5 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.load %6, %15 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
    ttg.local_store %8, %17 : tensor<16x16xf16, #blocked1> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
    %18 = ttg.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
    ttg.local_store %14, %18 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
    %19:7 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %6, %arg10 = %c0_i32, %arg11 = %17, %arg12 = %18, %arg13 = %16) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>, !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>, tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>) {
      %20 = arith.subi %arg1, %c2 : index
      %21 = arith.cmpi slt, %arg6, %20 : index
      %22 = arith.subi %arg1, %c1 : index
      %23 = arith.cmpi slt, %arg6, %22 : index
      %24 = ttg.local_load %arg11 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %25 = ttg.local_load %arg12 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %26 = tt.dot %24, %25, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %27 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %28 = tt.addptr %arg9, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %29 = tt.splat %23 : i1 -> tensor<16x16xi1, #blocked1>
      %30 = tt.load %27, %29 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %31 = tt.expand_dims %arg13 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %32 = tt.broadcast %31 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %33 = arith.muli %arg0, %32 : tensor<16x16xi64, #blocked>
      %34 = tt.addptr %arg5, %33 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %35 = tt.splat %23 : i1 -> tensor<16x16xi1, #blocked>
      %36 = tt.load %34, %35 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %37 = tt.splat %21 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
      %38 = tt.load %28, %37 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %39 = arith.addi %arg10, %c1_i32 : i32
      %40 = arith.cmpi slt, %39, %c1_i32 : i32
      %41 = arith.select %40, %39, %c0_i32 : i32
      %42 = ttg.memdesc_subview %0[%41, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
      ttg.local_store %30, %42 : tensor<16x16xf16, #blocked1> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
      %43 = ttg.memdesc_subview %1[%41, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
      ttg.local_store %36, %43 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>
      scf.yield %26, %27, %28, %41, %42, %43, %38 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>, !ttg.memdesc<16x16xf16, #shared2, #smem, mutable, 1x16x16>, tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    ttg.local_dealloc %0 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    tt.return %19#0 : tensor<16x16xf32, #mma>
  }
}

// -----

//   CHECK-LABEL: sink_convert_dealloc
// CHECK-COUNT-2: ttg.local_dealloc %{{.+}} : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//         CHECK: ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

//   CHECK-LABEL: anchor_barrier
//         CHECK: gpu.barrier
//         CHECK: tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @anchor_barrier(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked>) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    gpu.barrier
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = ttg.local_alloc %2 : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dont_hoist_scf_ops
  // Make sure we don't hoist scf ops above its dependencies.
  tt.func public @dont_hoist_scf_ops(%init: tensor<256x128xf32, #mfma>,
    %base: tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>,
    %p1: tensor<128x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>, %i1: i1) -> (tensor<256x128xf32, #mfma>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    // CHECK: scf.for
    %54 = scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %init) -> (tensor<256x128xf32, #mfma>)  : i32 {
      // CHECK: arith.addi
      %f = arith.addi %arg21, %c128_i32 : i32
      // CHECK: scf.if
      // CHECK: tt.load
      %p0 = scf.if %i1 -> tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>{
        %t = tt.splat %f : i32 -> tensor<256x128xi32>
        %padd = tt.addptr %base, %t : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>, tensor<256x128xi32>
        scf.yield %padd : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      } else {
        scf.yield %base : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      }
      %l = tt.load %p0 : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %r = tt.load %p1 : tensor<128x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %acc = tt.dot %l, %r, %arg : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      scf.yield %acc : tensor<256x128xf32, #mfma>
    }
    tt.return %54 : tensor<256x128xf32, #mfma>
  }
}
