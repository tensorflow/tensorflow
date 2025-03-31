// RUN: triton-opt %s -split-input-file -tritongpu-pipeline -canonicalize | FileCheck --dump-input-context=50 %s
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=CHECK-NOCANON

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #ttg.slice<{parent=#AL, dim=0}>
#BLs0 = #ttg.slice<{parent=#BL, dim=0}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @matmul_loop
// CHECK-DAG: %[[CONSTANT_NEG1:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK: %[[ABUFFER:.*]] = ttg.local_alloc
// CHECK: %[[BBUFFER:.*]] = ttg.local_alloc
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[ASUB:.*]] = ttg.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32>
// CHECK: %[[T_A0:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[ASUB]] mask %[[LOOP_COND_0_SPLAT_A]] : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable, 2x128x32>
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[BSUB:.*]] = ttg.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B0:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[BSUB]] mask %[[LOOP_COND_0_SPLAT_B]] other %{{.*}} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable, 2x32x128>
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[ASUB1:.*]] = ttg.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_A1:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[ASUB1]] mask %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[BSUB1:.*]] = ttg.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B1:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[BSUB1]] mask %[[LOOP_COND_1_SPLAT_B]]
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_NEG1]]
// CHECK:   %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK:   %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK:   %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK:   ttg.async_wait {{.*}} {num = 2 : i32}
// CHECK:   %[[A:.*]] = ttg.memdesc_subview %[[ABUFFER]][%[[EXT_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[arg_a0_dot_op:.*]] = ttg.local_load %[[A]]
// CHECK:   %[[B:.*]] = ttg.memdesc_subview %[[BBUFFER]][%[[EXT_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = ttg.local_load %[[B]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_0]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   %[[ASUB3:.*]] = ttg.memdesc_subview %[[ABUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = ttg.async_copy_global_to_local {{.*}}, %[[ASUB3]]
// CHECK:   %[[BSUB3:.*]] = ttg.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = ttg.async_copy_global_to_local {{.*}}, %[[BSUB3]]
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                       %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                       %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>


  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: dot_chained_single_load
  tt.func @dot_chained_single_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x64xf32, #mma> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   ttg.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   ttng.warp_group_dot
    // CHECK-NEXT: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
    // CHECK:   ttng.warp_group_dot
    // CHECK:   ttg.async_copy_global_to_local
    // CHECK:   ttg.async_commit_group
    // CHECK:   scf.yield
    %17:2 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_3, %arg5 = %16) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>)  : i32 {
      %18 = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %21 = ttng.warp_group_dot %19, %20, %cst_2 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %23 = ttg.memdesc_trans %20 {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared1, #smem> -> !ttg.memdesc<16x64xf16, #shared, #smem>
      %24 = ttg.convert_layout %22 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
      %25 = ttng.warp_group_dot %24, %23, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>> * !ttg.memdesc<16x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %25, %26 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>
    }
    tt.return %17#0 : tensor<128x64xf32, #mma>
  }

  // Check that we are able to perform WGMMA pipelining if the accumulator is conditionally being modified
  // CHECK-LABEL: dot_acc_cond_modified
  tt.func @dot_acc_cond_modified(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext : i32) -> tensor<128x16xf32, #mma1> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %2 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   ttg.async_wait {{.*}} {num = 2 : i32}
    // CHECK:   ttng.warp_group_dot
    // CHECK-NEXT: ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32}
    // CHECK:   ttg.async_copy_global_to_local
    // CHECK:   ttg.async_commit_group
    // CHECK:   scf.if
    // CHECK:     ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
    // CHECK:     arith.mulf
    // CHECK:     scf.yield
    // CHECK:   scf.yield
    // CHECK:   ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %16, %arg6 = %8) -> (tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %9 = tt.load %arg6 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %18 = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %acc = ttng.warp_group_dot %19, %20, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        %acc_zero = arith.mulf %acc, %cst_2 : tensor<128x16xf32, #mma1>
        scf.yield %acc_zero : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      %22 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %23 = tt.addptr %arg6, %cst2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %acc_, %22, %23 : tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: two_accumulator_escape
  tt.func @two_accumulator_escape(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>) {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
    // CHECK: %[[ALLOC1:.+]] = ttg.local_alloc
    // CHECK: %[[ALLOC2:.+]] = ttg.local_alloc
    // CHECK: %[[R:.+]]:{{.+}} = scf.for
    // CHECK:   %[[DOT1:.+]] = ttng.warp_group_dot{{.*}}
    // CHECK:   ttg.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   %[[TRANS:.+]] = ttg.memdesc_trans{{.*}} : !ttg.memdesc
    // CHECK:   %[[DOT2:.+]] = ttng.warp_group_dot{{.*}} %[[TRANS]]
    // CHECK:   ttng.warp_group_dot_wait %[[DOT1]], %[[DOT2]], %[[ALLOC1]], %[[ALLOC2]], %[[TRANS]] {pendings = 2 : i32}
    // CHECK:   scf.yield
    // CHECK: %{{.*}}:2 = ttng.warp_group_dot_wait %[[R]]#{{.+}}, %[[R]]#{{.+}} {pendings = 0 : i32} : tensor<128x16xf32, #{{.*}}>, tensor<128x64xf32, #{{.*}}>
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_3, %arg5 = %16, %arg6 = %cst_2) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>)  : i32 {
      %21 = ttng.warp_group_dot %19, %20, %arg6 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %l = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %c = ttg.local_alloc %l : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %23 = ttg.memdesc_trans %c {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared1, #smem> -> !ttg.memdesc<16x64xf16, #shared, #smem>
      %25 = ttng.warp_group_dot %cst_4, %23, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>> * !ttg.memdesc<16x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %25, %26, %21 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>
    }
    tt.return %17#0, %17#2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory

// Make sure that if one of the load dot operand is not pipelined (and therefore not double buffered) we won't use
// async dot.
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: no_wgmma_pipeline
  tt.func public @no_wgmma_pipeline(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<512> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_2 = arith.constant dense<512> : tensor<128x1xi32, #blocked>
    %cst_3 = arith.constant dense<512> : tensor<128x1xi32, #blocked1>
    %cst_4 = arith.constant dense<512> : tensor<64x1xi32, #blocked1>
    %cst_5 = arith.constant dense<32768> : tensor<64x256xi32, #blocked1>
    %cst_6 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = arith.remsi %0, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.remsi %2, %cst_1 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = arith.muli %4, %cst_2 : tensor<128x1xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %9 = tt.broadcast %7 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %10 = arith.addi %8, %9 : tensor<128x64xi32, #blocked>
    %11 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<128x64x!tt.ptr<f8E5M2>, #blocked>
    %12 = tt.addptr %11, %10 : tensor<128x64x!tt.ptr<f8E5M2>, #blocked>, tensor<128x64xi32, #blocked>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %15 = arith.muli %14, %cst_4 : tensor<64x1xi32, #blocked1>
    %16 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %17 = tt.broadcast %15 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %18 = tt.broadcast %16 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %19 = arith.addi %17, %18 : tensor<64x256xi32, #blocked1>
    %20 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>
    %21 = tt.addptr %20, %19 : tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<64x256xi32, #blocked1>
    %22:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %12, %arg6 = %21) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f8E5M2>, #blocked>, tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>)  : i32 {
      %35 = tt.load %arg5 : tensor<128x64x!tt.ptr<f8E5M2>, #blocked>
      %36 = tt.load %arg6 : tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>
      %37 = ttg.local_alloc %35 : (tensor<128x64xf8E5M2, #blocked>) -> !ttg.memdesc<128x64xf8E5M2, #shared, #smem>
      %38 = ttg.local_alloc %36 : (tensor<64x256xf8E5M2, #blocked1>) -> !ttg.memdesc<64x256xf8E5M2, #shared1, #smem>
      // CHECK: ttg.local_alloc
      // CHECK: scf.for
      // CHECK:   ttng.warp_group_dot
      // CHECK-NEXT: ttng.warp_group_dot_wait
      %39 = ttng.warp_group_dot %37, %38, %arg4 {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x64xf8E5M2, #shared, #smem> * !ttg.memdesc<64x256xf8E5M2, #shared1, #smem> -> tensor<128x256xf32, #mma>
      %40 = tt.addptr %arg5, %cst_6 : tensor<128x64x!tt.ptr<f8E5M2>, #blocked>, tensor<128x64xi32, #blocked>
      %41 = tt.addptr %arg6, %cst_5 : tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<64x256xi32, #blocked1>
      scf.yield %39, %40, %41 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f8E5M2>, #blocked>, tensor<64x256x!tt.ptr<f8E5M2>, #blocked1>
    }
    %23 = arith.truncf %22#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %25 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %26 = arith.muli %25, %cst_3 : tensor<128x1xi32, #blocked1>
    %27 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<128x1x!tt.ptr<f8E5M2>, #blocked1>
    %28 = tt.addptr %27, %26 : tensor<128x1x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x1xi32, #blocked1>
    %29 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %30 = tt.broadcast %28 : tensor<128x1x!tt.ptr<f8E5M2>, #blocked1> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
    %31 = tt.broadcast %29 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %32 = tt.addptr %30, %31 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
    %33 = tt.fp_to_fp %23 {rounding = 1 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf8E5M2, #mma>
    %34 = ttg.convert_layout %33 : tensor<128x256xf8E5M2, #mma> -> tensor<128x256xf8E5M2, #blocked1>
    tt.store %32, %34 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
    tt.return
  }
}

// -----

// A dot can be properly async if all its uses follow a synchronous MMAv3 dot.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: async_following_sync
  tt.func @async_following_sync(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>) {
    %cst = arith.constant dense<64> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32

    // Add a "dummy" early return here to test that we don't crash in the
    // presence of unstructured control flow.
    %cond = arith.constant 0 : i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %zero = arith.constant 0.0 : f32
    %t1 = tt.splat %zero : f32 -> tensor<128x64xf32, #mma>
    %t2 = tt.splat %zero : f32 -> tensor<128x16xf32, #mma1>
    tt.return %t1, %t2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  ^bb2:  // pred: ^bb0

    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
    // CHECK:          %[[LOOP:[^ :]+]]{{.*}} scf.for {{.*}} iter_args(%[[PREV_DOT2:[^ ]+]]
    // CHECK-NOT:        ttng.warp_group_dot_wait
    // CHECK:            %[[DOT0:.+]] = ttng.warp_group_dot
    // CHECK-NOT:        ttng.warp_group_dot_wait
    // CHECK:            %[[DOT1:.+]] = ttng.warp_group_dot
    // CHECK-NEXT:       ttng.warp_group_dot_wait
    // CHECK-DAG-SAME:     %[[DOT0]]
    // CHECK-DAG-SAME:     %[[DOT1]]
    // CHECK-DAG-SAME:     %[[PREV_DOT2]]
    // CHECK-SAME:         {pendings = 0 : i32}
    // CHECK:            %[[DOT2:.+]] = ttng.warp_group_dot
    // CHECK-NOT:        ttng.warp_group_dot_wait
    // CHECK:          scf.yield %[[DOT2]]
    // CHECK:          ttng.warp_group_dot_wait %[[LOOP]]#3, %[[LOOP]]#0 {pendings = 0 : i32}
    %17:4 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%prev_dot2 = %cst_3, %arg5 = %16, %prev_dot1 = %cst_2, %prev_dot0 = %cst_2) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>)  : i32 {
      // This one can be async.
      %dot0 = ttng.warp_group_dot %19, %20, %prev_dot1 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      // This can't be async because its result is modified before it's yielded.
      %dot1 = ttng.warp_group_dot %19, %20, %prev_dot1 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %dot1.1 = arith.addf %dot1, %dot1 : tensor<128x16xf32, #mma1>
      %l = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %c = ttg.local_alloc %l : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %23 = ttg.memdesc_trans %c {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared1, #smem> -> !ttg.memdesc<16x64xf16, #shared, #smem>
      // This dot can be async even though %prev_dot2 is not used directly by an
      // async dot, because that use follows the synchronous dot above.
      %prev_dot2.1 = arith.addf %prev_dot2, %prev_dot2 : tensor<128x64xf32, #mma>
      %dot2 = ttng.warp_group_dot %cst_4, %23, %prev_dot2.1 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>> * !ttg.memdesc<16x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %dot2, %26, %dot1.1, %dot0 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>
    }
    tt.return %17#0, %17#2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  }
}

// -----
// Test pipelining of descriptor_store
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tma_store_pipeline
  tt.func public @tma_store_pipeline(%arg0: tensor<1xf32, #blocked>, %arg1: !tt.tensordesc<tensor<1xf32, #shared>>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg4 = %c0_i32 to %arg3 step %arg2  : i32 {
      %1 = arith.divsi %arg4, %arg2 : i32
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      // CHECK-NEXT: ttg.local_store
      // CHECK-NEXT: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tensor_desc_to_tma_ptr
      // CHECK-NEXT: ttng.async_tma_copy_local_to_global
      tt.descriptor_store %arg1[%1], %arg0 : !tt.tensordesc<tensor<1xf32, #shared>>, tensor<1xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tma_store_pipeline
  tt.func public @tma_store_pipeline(%arg0: tensor<8x128xf32, #blocked>, %arg1: !tt.tensordesc<tensor<1x128xf32, #shared>>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg4 = %c0_i32 to %arg3 step %arg2  : i32 {
      %1 = arith.divsi %arg4, %arg2 : i32
      %2 = tt.splat %1 : i32 -> tensor<8xi32, #blocked1>
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      // CHECK-NEXT: ttg.local_store
      // CHECK-NEXT: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tensor_desc_to_tma_ptr
      // CHECK-NEXT: ttng.async_tma_scatter
      tt.descriptor_scatter %arg1[%2, %1], %arg0 : !tt.tensordesc<tensor<1x128xf32, #shared>>, tensor<8xi32, #blocked1>, i32, tensor<8x128xf32, #blocked>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tma_multiple_store_pipeline
  tt.func public @tma_multiple_store_pipeline(%arg0: tensor<1xf32, #blocked>, %arg1: !tt.tensordesc<tensor<1xf32, #shared>>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %[[ALLOC:.+]] = ttg.local_alloc : () -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    // CHECK: scf.for
    scf.for %arg4 = %c0_i32 to %arg3 step %arg2  : i32 {
      %1 = arith.divsi %arg4, %arg2 : i32
      %2 = arith.divsi %arg2, %arg4 : i32
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      // CHECK-NEXT: ttg.local_store %{{.+}}, %[[ALLOC]]
      // CHECK-NEXT: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tensor_desc_to_tma_ptr
      // CHECK-NEXT: ttng.async_tma_copy_local_to_global %{{.*}} %[[ALLOC]]
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      // CHECK-NEXT: ttg.local_store %{{.+}}, %[[ALLOC]]
      // CHECK-NEXT: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tensor_desc_to_tma_ptr
      // CHECK-NEXT: ttng.async_tma_copy_local_to_global %{{.*}} %[[ALLOC]]
      tt.descriptor_store %arg1[%1], %arg0 : !tt.tensordesc<tensor<1xf32, #shared>>, tensor<1xf32, #blocked>
      tt.descriptor_store %arg1[%2], %arg0 : !tt.tensordesc<tensor<1xf32, #shared>>, tensor<1xf32, #blocked>
    }
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: _kernel_matmul_dependency
  tt.func public @_kernel_matmul_dependency(%arg0: tensor<128x128x!tt.ptr<f8E4M3FNUZ>, #blocked>, %arg1: !tt.ptr<f8E4M3FNUZ> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %1 = tt.splat %arg1 : !tt.ptr<f8E4M3FNUZ> -> tensor<128x128x!tt.ptr<f8E4M3FNUZ>, #blocked1>
    %2:4 = scf.for %arg6 = %c8_i32 to %arg3 step %c8_i32 iter_args(%arg7 = %c8_i32, %arg8 = %c8_i32, %arg9 = %cst_1, %arg10 = %arg5) -> (i32, i32, tensor<128x128xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>)  : i32 {
      %3 = arith.addi %arg7, %c8_i32 : i32
      %4 = arith.cmpi eq, %3, %c8_i32 : i32
      %5:2 = scf.if %4 -> (i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) {
        %21 = arith.addi %arg8, %c8_i32 : i32
        scf.yield %21, %arg5 : i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      } else {
        scf.yield %arg8, %arg10 : i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      }
      %6 = arith.cmpi eq, %3, %c8_i32 : i32
      %7 = scf.if %6 -> (f32) {
        scf.yield %cst_0 : f32
      } else {
        %21 = tt.load %arg4 : !tt.ptr<f32>
        scf.yield %21 : f32
      }
      %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %9 = arith.addi %8, %0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %11 = tt.broadcast %10 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
      %12 = tt.addptr %1, %11 : tensor<128x128x!tt.ptr<f8E4M3FNUZ>, #blocked1>, tensor<128x128xi32, #blocked1>
      %13 = tt.load %arg0 : tensor<128x128x!tt.ptr<f8E4M3FNUZ>, #blocked>
      %14 = ttg.local_alloc %13 : (tensor<128x128xf8E4M3FNUZ, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FNUZ, #shared, #smem>
      %15 = tt.load %12 : tensor<128x128x!tt.ptr<f8E4M3FNUZ>, #blocked1>
      %16 = ttg.local_alloc %15 : (tensor<128x128xf8E4M3FNUZ, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FNUZ, #shared1, #smem>
      %17 = ttng.warp_group_dot %14, %16, %arg9 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x128xf8E4M3FNUZ, #shared, #smem> * !ttg.memdesc<128x128xf8E4M3FNUZ, #shared1, #smem> -> tensor<128x128xf32, #mma>
      %18 = tt.splat %7 : f32 -> tensor<128x128xf32, #mma>
      %19 = arith.mulf %17, %18 : tensor<128x128xf32, #mma>
      %20 = scf.if %6 -> (tensor<128x128xf32, #mma>) {
        scf.yield %cst_1 : tensor<128x128xf32, #mma>
      } else {
        scf.yield %19 : tensor<128x128xf32, #mma>
      }
      scf.yield %3, %5#0, %20, %5#1 : i32, i32, tensor<128x128xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    }
    tt.return
  }
}

// -----

// Pipeline the if ops at the beginning and the end of the loop
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: dot_prologue_epilogue
  // COMMON: {{.*}}, {{.*}}, %[[EXT:.*]]: i32, {{.*}}
  tt.func @dot_prologue_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %2 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // COMMON: %[[C0:.*]] = arith.constant 0 : i32
    // COMMON: scf.for %[[IND_VAR:.*]] = %[[C0]]
    // COMMON-NOT: load
    // COMMON: %[[CND:.*]] = arith.cmpi slt, %[[IND_VAR]], %[[EXT]]
    // COMMON: scf.if %[[CND]]
    // COMMON: dot
    // COMMON: scf.if %[[CND]]
    // COMMON:   arith.mulf
    // COMMON:   scf.yield
    // COMMON-NOT: tt.addptr
    // COMMON: scf.yield
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %16, %arg6 = %8) -> (tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %9 = tt.load %arg6 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %inc_ptr = scf.if %cnd -> tensor<64x16x!tt.ptr<f16>, #blocked> {
        %ptr = tt.addptr %arg5, %inc : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
        scf.yield %ptr : tensor<64x16x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      }
      %18 = tt.load %inc_ptr : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %acc = ttng.warp_group_dot %19, %20, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        %acc_zero = arith.mulf %acc, %cst_2 : tensor<128x16xf32, #mma1>
        scf.yield %acc_zero : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      %22 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %23 = tt.addptr %arg6, %cst2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %acc_, %22, %23 : tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma1>
  }
}

// -----

// Verify that uses of the ops scheduled in partucular place of the loop (like epilogue if) are correctly scheduled too.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-NOCANON-LABEL: pipeline_downstream_dependencies
  // CHECK-NOCANON: {{.*}}, {{.*}}, %[[EXT:.*]]: i32, {{.*}}
  tt.func @pipeline_downstream_dependencies(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst1 = arith.constant dense<1> : tensor<64x16xi32, #blocked>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %2 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK-NOCANON: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK-NOCANON: scf.for %[[IND_VAR:.*]] = %[[C0]]
    // CHECK-NOCANON-NOT load
    // CHECK-NOCANON: dot
    // CHECK-NOCANON: %[[CND:.*]] = arith.cmpi slt, %[[IND_VAR]], %[[EXT]]
    // CHECK-NOCANON: %[[IFRET:.*]]:2 = scf.if %[[CND]]
    // CHECK-NOCANON:   arith.mulf
    // CHECK-NOCANON:   scf.yield
    // CHECK-NOCANON: tt.addptr {{.*}}, %[[IFRET]]#1
    // CHECK-NOCANON: scf.yield
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %16, %arg6 = %8) -> (tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %9 = tt.load %arg6 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %18 = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %20 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %acc = ttng.warp_group_dot %19, %20, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %if_ret:2 = scf.if %cnd -> (tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>) {
        %acc_zero = arith.mulf %acc, %cst_2 : tensor<128x16xf32, #mma1>
        scf.yield %acc_zero, %cst : tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>
      } else {
        scf.yield %acc, %cst1 : tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>
      }
      %22 = tt.addptr %arg5, %if_ret#1 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %23 = tt.addptr %arg6, %cst2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %if_ret#0, %22, %23 : tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: dot_lhs_registers
  tt.func @dot_lhs_registers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_3 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %cst_4 = arith.constant dense<2.0> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   ttg.async_wait {{.*}} {num = 2 : i32}
    // CHECK:   ttg.local_load
    // CHECK:   ttng.warp_group_dot
    // CHECK-NEXT: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
    // CHECK:   ttg.async_copy_global_to_local
    // CHECK:   ttg.async_commit_group
    // CHECK:   ttg.async_copy_global_to_local
    // CHECK:   ttg.async_commit_group
    // CHECK:   scf.yield
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %8, %arg6 = %16) -> (tensor<128x16xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>,
        tensor<64x16x!tt.ptr<f16>, #blocked>)  : i32 {
      %a_block = tt.load %arg5 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %b_block = tt.load %arg6 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %a_dotop = ttg.convert_layout %a_block : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %a_dotop_mul = arith.mulf %a_dotop, %cst_4 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %b_smem = ttg.local_alloc %b_block : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared1, #smem>
      %21 = ttng.warp_group_dot %a_dotop_mul, %b_smem, %arg4 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma>
      %25 = tt.addptr %arg5, %cst_3 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %26 = tt.addptr %arg6, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %21, %25, %26 : tensor<128x16xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x16x!tt.ptr<f16>, #blocked>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma>
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 32]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mmav3_fp8_row_major_rhs(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: mmav3_fp8_row_major_rhs
    // The col-major RHS SMEM encoding in the input, created by accelerate-matmul, should be overwritten by the row-major TMA layout.
    // Note that this "overwriting" makes the program invalid after SWP, since warp_group_dot does not support row-major fp8 RHS.
    // In this case, the TMA load on B should not be pipelined. When this bug is fixed, this test should be rewritten to verify that.
    // CHECK-NOT: order = [0, 1]
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = arith.muli %4, %c64_i32 : i32
    %7 = arith.addi %arg5, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared>>
    %10 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf8E4M3FN, #shared>>
    %true = arith.constant true
    %false = arith.constant false
    %11:2 = scf.for %arg6 = %c0_i32 to %8 step %c1_i32 iter_args(%arg7 = %cst, %arg8 = %c0_i32) -> (tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 32]}>>, i32)  : i32 {
      %14 = tt.descriptor_load %9[%5, %arg8] : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared>> -> tensor<128x64xf8E4M3FN, #blocked>
      %15 = ttg.local_alloc %14 : (tensor<128x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>
      %16 = tt.descriptor_load %10[%arg8, %6] : !tt.tensordesc<tensor<64x64xf8E4M3FN, #shared>> -> tensor<64x64xf8E4M3FN, #blocked>
      %17 = ttg.local_alloc %16 : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared1, #ttg.shared_memory>
      %18 = ttng.warp_group_dot %15, %17, %arg7 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory> * !ttg.memdesc<64x64xf8E4M3FN, #shared1, #ttg.shared_memory> -> tensor<128x64xf32, #mma>
      %19 = arith.addi %arg8, %c64_i32 : i32
      scf.yield %18, %19 : tensor<128x64xf32, #mma>, i32
    }
    %12 = ttg.convert_layout %11#0 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    %13 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x64xf32, #nvmma_128>>
    tt.descriptor_store %13[%5, %6], %12 : !tt.tensordesc<tensor<128x64xf32, #nvmma_128>>, tensor<128x64xf32, #blocked>
    tt.return
  }
}
