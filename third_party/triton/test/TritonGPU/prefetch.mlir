// RUN: triton-opt %s -split-input-file -tritongpu-prefetch -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @matmul_loop_mixed
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[A0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[B0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C16]]]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C16]], %[[C0]]]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x32xf8E5M2, #A, #smem> -> tensor<128x32xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x32xf8E5M2, #A_OP> -> tensor<128x32xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<32x128xf16, #B, #smem> -> tensor<32x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

// 4 warps
// matmul: 128x16 @ 16x128 -> 128x128
// CHECK: tt.func @matmul_loop_mixed
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[A0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[B0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[a0_prefetch]], %[[b0_prefetch]], {{.*}}
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x16x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x16xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x16xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<16x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<16x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x16xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<16x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x16x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x16xf8E5M2, #AL>) -> !ttg.memdesc<128x16xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<16x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<16x128xf16, #BL>) -> !ttg.memdesc<16x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<16x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x16xf8E5M2, #A, #smem>, !ttg.memdesc<16x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x16xf8E5M2, #A, #smem> -> tensor<128x16xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x16xf8E5M2, #A_OP> -> tensor<128x16xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<16x128xf16, #B, #smem> -> tensor<16x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x16xf16, #A_OP> * tensor<16x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<128x16xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<16x128x!tt.ptr<f16>, #BL>, tensor<16x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x16x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x16xf8E5M2, #AL>) -> !ttg.memdesc<128x16xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<16x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<16x128xf16, #BL>) -> !ttg.memdesc<16x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<16x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x16xf8E5M2, #A, #smem>, !ttg.memdesc<16x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

#AL_3D = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>
#BL_3D = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>
#A_3D = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [2, 0, 1]}>
#B_3D = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [2, 0, 1]}>
#C_3D = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 4, 1]}>
#A_OP_3D = #ttg.dot_op<{opIdx = 0, parent = #C_3D, kWidth = 2}>
#B_OP_3D = #ttg.dot_op<{opIdx = 1, parent = #C_3D, kWidth = 2}>

// matmul: 8x128x16 @ 8x16x128 -> 8x128x128
// CHECK: tt.func @matmul_3D_loop_mixed
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[A0:.*]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[B0:.*]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[a0_prefetch]], %[[b0_prefetch]], {{.*}}
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_3D_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<8x128x128xf32, #C_3D>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<8x16x128x!tt.ptr<f16>, #BL_3D>

  %a_mask = arith.constant dense<true> : tensor<8x128x16xi1, #AL_3D>
  %a_other = arith.constant dense<0.00e+00> : tensor<8x128x16xf8E5M2, #AL_3D>
  %b_mask = arith.constant dense<true> : tensor<8x16x128xi1, #BL_3D>
  %b_other = arith.constant dense<0.00e+00> : tensor<8x16x128xf16, #BL_3D>
  %c_init = arith.constant dense<0.00e+00> : tensor<8x128x128xf32, #C_3D>

  %a_off = arith.constant dense<4> : tensor<8x128x16xi32, #AL_3D>
  %b_off = arith.constant dense<4> : tensor<8x16x128xi32, #BL_3D>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
  %a_init = ttg.local_alloc %a_ : (tensor<8x128x16xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>
  %b_init = ttg.local_alloc %b_ : (tensor<8x16x128xf16, #BL_3D>) -> !ttg.memdesc<8x16x128xf16, #B_3D, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x16x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem> -> tensor<8x128x16xf8E5M2, #A_OP_3D>
    %a_op = tt.fp_to_fp %a_op_ : tensor<8x128x16xf8E5M2, #A_OP_3D> -> tensor<8x128x16xf16, #A_OP_3D>
    %b_op = ttg.local_load %b : !ttg.memdesc<8x16x128xf16, #B_3D, #smem> -> tensor<8x16x128xf16, #B_OP_3D>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<8x128x16xf16, #A_OP_3D> * tensor<8x16x128xf16, #B_OP_3D> -> tensor<8x128x128xf32, #C_3D>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x128x16xi32, #AL_3D>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, tensor<8x16x128xi32, #BL_3D>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
    %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x16xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>
    %next_b = ttg.local_alloc %b_ : (tensor<8x16x128xf16, #BL_3D>) -> !ttg.memdesc<8x16x128xf16, #B_3D, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x16x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>
  }
  tt.return %loop#4 : tensor<8x128x128xf32, #C_3D>
}
}  // end module

// matmul: 8x128x32 @ 8x32x128 -> 8x128x128
// CHECK: tt.func @matmul_3D_loop_mixed2
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[A0:.*]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[B0:.*]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]], %[[C16]]]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C16]], %[[C0]]]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_3D_loop_mixed2(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<8x128x128xf32, #C_3D>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<8x32x128x!tt.ptr<f16>, #BL_3D>

  %a_mask = arith.constant dense<true> : tensor<8x128x32xi1, #AL_3D>
  %a_other = arith.constant dense<0.00e+00> : tensor<8x128x32xf8E5M2, #AL_3D>
  %b_mask = arith.constant dense<true> : tensor<8x32x128xi1, #BL_3D>
  %b_other = arith.constant dense<0.00e+00> : tensor<8x32x128xf16, #BL_3D>
  %c_init = arith.constant dense<0.00e+00> : tensor<8x128x128xf32, #C_3D>

  %a_off = arith.constant dense<4> : tensor<8x128x32xi32, #AL_3D>
  %b_off = arith.constant dense<4> : tensor<8x32x128xi32, #BL_3D>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
  %a_init = ttg.local_alloc %a_ : (tensor<8x128x32xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>
  %b_init = ttg.local_alloc %b_ : (tensor<8x32x128xf16, #BL_3D>) -> !ttg.memdesc<8x32x128xf16, #B_3D, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x32x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem> -> tensor<8x128x32xf8E5M2, #A_OP_3D>
    %a_op = tt.fp_to_fp %a_op_ : tensor<8x128x32xf8E5M2, #A_OP_3D> -> tensor<8x128x32xf16, #A_OP_3D>
    %b_op = ttg.local_load %b : !ttg.memdesc<8x32x128xf16, #B_3D, #smem> -> tensor<8x32x128xf16, #B_OP_3D>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<8x128x32xf16, #A_OP_3D> * tensor<8x32x128xf16, #B_OP_3D> -> tensor<8x128x128xf32, #C_3D>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x128x32xi32, #AL_3D>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, tensor<8x32x128xi32, #BL_3D>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
    %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x32xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>
    %next_b = ttg.local_alloc %b_ : (tensor<8x32x128xf16, #BL_3D>) -> !ttg.memdesc<8x32x128xf16, #B_3D, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x32x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>
  }
  tt.return %loop#4 : tensor<8x128x128xf32, #C_3D>
}
}  // end module

// CHECK: tt.func @matmul_loop_yield_no_operand
// CHECK: scf.for
// CHECK: scf.if
// CHECK: tt.store
// CHECK-NOT: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_loop_yield_no_operand(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.addi %arg8, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.addi %0, %c31_i32 : i32
    %4 = arith.divsi %3, %c32_i32 : i32
    %5 = arith.muli %1, %4 : i32
    %6 = tt.get_program_id x : i32
    %7 = tt.get_num_programs x : i32
    %8 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    scf.for %arg11 = %6 to %5 step %7  : i32 {
      %9 = arith.divsi %arg11, %4 : i32
      %10 = arith.remsi %9, %2 : i32
      %11 = tt.load %8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %12 = tt.load %8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %13 = ttg.convert_layout %12 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %14 = ttg.convert_layout %11 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = tt.dot %13, %14, %cst, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %16 = arith.cmpi sgt, %10, %c0_i32 : i32
      %17 = scf.if %16 -> (tensor<32x32xf32, #mma>) {
        %21 = tt.dot %13, %14, %15, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
        scf.yield %21 : tensor<32x32xf32, #mma>
      } else {
        scf.yield %15 : tensor<32x32xf32, #mma>
      }
      %18 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
      %19 = arith.truncf %17 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
      %20 = ttg.convert_layout %19 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked1>
      tt.store %18, %20 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    tt.return
  }
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = false}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @matmul_loop_mixed_amd
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[A0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subview %[[B0:.*]][%[[C0]], %[[C0]]]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C16]]]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C16]], %[[C0]]]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subview {{.*}}[%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed_amd(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x32xf8E5M2, #A, #smem> -> tensor<128x32xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x32xf8E5M2, #A_OP> -> tensor<128x32xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<32x128xf16, #B, #smem> -> tensor<32x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

 // -----

// CHECK: tt.func @matmul_loop_on_blocked_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_loop_on_blocked_layout(%arg_lhs: !ttg.memdesc<16x512xf32, #shared, #smem, mutable>, %arg_rhs: !ttg.memdesc<512x32xf32, #shared, #smem, mutable>, %arg_init: tensor<16x32xf32, #blocked>, %itr_val : i32) -> (tensor<16x32xf32, #blocked>) {
    %loop:3 = scf.for %itr = %itr_val to %itr_val step %itr_val iter_args(%init = %arg_init, %lhs = %arg_lhs, %rhs = %arg_rhs) -> (tensor<16x32xf32, #blocked>, !ttg.memdesc<16x512xf32, #shared, #smem, mutable>, !ttg.memdesc<512x32xf32, #shared, #smem, mutable>) : i32 {
      %lhs_ll = ttg.local_load %lhs : !ttg.memdesc<16x512xf32, #shared, #smem, mutable> -> tensor<16x512xf32, #blocked>
      %lhs_ll_cvt = ttg.convert_layout %lhs_ll : tensor<16x512xf32, #blocked> -> tensor<16x512xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %rhs_ll = ttg.local_load %rhs : !ttg.memdesc<512x32xf32, #shared, #smem, mutable> -> tensor<512x32xf32, #blocked>
      %rhs_ll_cvt = ttg.convert_layout %rhs_ll : tensor<512x32xf32, #blocked> -> tensor<512x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %res = tt.dot %lhs_ll_cvt, %rhs_ll_cvt, %init : tensor<16x512xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<512x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x32xf32, #blocked>
      scf.yield %res, %lhs, %rhs : tensor<16x32xf32, #blocked>, !ttg.memdesc<16x512xf32, #shared, #smem, mutable>, !ttg.memdesc<512x32xf32, #shared, #smem, mutable>
    }
    tt.return %loop#0 : tensor<16x32xf32, #blocked>
  }
} // end module
