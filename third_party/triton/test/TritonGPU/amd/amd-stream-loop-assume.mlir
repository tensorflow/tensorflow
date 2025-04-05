// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=2" -canonicalize | FileCheck %s

// matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #ttg.slice<{parent=#AL, dim=0}>
#BLs0 = #ttg.slice<{parent=#BL, dim=0}>
#BLs1 = #ttg.slice<{parent=#BL, dim=1}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK-LABEL: tt.func @assume_matmul
// CHECK-COUNT-2: tt.load
// CHECK-COUNT-2: ttg.local_store
// CHECK: scf.for
// CHECK: llvm.intr.assume
// CHECK: tt.load
// CHECK: ttg.local_load
// CHECK: tt.load
// CHECK: ttg.local_load
// CHECK: tt.dot
// CHECK-COUNT-2: ttg.local_store
// CHECK: scf.yield
// CHECK: llvm.intr.assume
// CHECK-COUNT-2: ttg.local_load
// CHECK: tt.dot
// CHECK-NOT: tt.dot

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @assume_matmul(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
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

  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>
  %c_true = arith.constant 1: i1

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // Note: This isn't a meaningful assumption here, but it acts
    // as a placeholder for a user generated assume in a loop.
    llvm.intr.assume %c_true : i1
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_ = ttg.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}
}
