// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefix=CHECK --check-prefix=CF
// RUN: triton-opt %s -split-input-file                     --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefix=CHECK --check-prefix=SCF
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefix=CHECK --check-prefix=CF
// RUN: triton-opt %s -split-input-file                     --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefix=CHECK --check-prefix=SCF

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #ttg.slice<{dim = 0, parent = #AL}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A_DOT = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_DOT = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK-LABEL: matmul_loop
// There shouldn't be any membar with the dot op encoding.
tt.func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B_DOT>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}

// CHECK-LABEL: raw_single_block
tt.func @raw_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 : tensor<128x32x!tt.ptr<f16>, #AL>
  %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %3 = ttg.local_load %2 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: war_single_block
tt.func @war_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 : tensor<128x32x!tt.ptr<f16>, #AL>
  %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: ttg.local_alloc
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %3 = ttg.local_load %2 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: %4 = ttg.local_alloc
  %4 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  tt.return
}

// CHECK-LABEL: war_single_block_local_store
tt.func @war_single_block_local_store(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 : tensor<128x32x!tt.ptr<f16>, #AL>
  %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %3 = ttg.local_load %2 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_store
  ttg.local_store %1, %2 : tensor<128x32xf16, #AL> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

// CHECK-LABEL: scratch
tt.func @scratch(%arg: tensor<16x16xf16, #AL>) {
  %cst0 = ttg.local_alloc %arg : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  // CHECK: gpu.barrier
  // CHECK: tt.reduce
  %1 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  %2 = "tt.reduce" (%1) ({
  ^bb0(%arg1: f16, %arg2: f16):
    %add = arith.addf %arg1, %arg2 : f16
    tt.reduce.return %add : f16
  }) {axis = 0 : i32} : (tensor<16x16xf16, #AL>) -> tensor<16xf16, #sliceAd0>
  tt.return
}

// CHECK-LABEL: async_wait
tt.func @async_wait(%arg: tensor<32x16xf16, #AL>) {
  %cst0 = ttg.local_alloc %arg : (tensor<32x16xf16, #AL>) -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: ttg.async_wait
  ttg.async_wait {num = 4 : i32}
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %1 = ttg.local_load %cst0 : !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<32x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: subview
tt.func @subview() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #AL>
  %a = ttg.local_alloc %cst0 : (tensor<32x16xf16, #AL>) -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory>
  %index = arith.constant 0 : i32
  %0 = ttg.memdesc_subview %a[%index, %index] : !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_alloc
  %2 = ttg.local_alloc %1 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  tt.return
}

// CHECK-LABEL: trans
tt.func @trans(%a: !ttg.memdesc<16x32xf16, #A_SHARED, #ttg.shared_memory>) {
  // CHECK-NOT: gpu.barrier
  %b = ttg.memdesc_trans %a {order=array<i32: 1,0>} : !ttg.memdesc<16x32xf16, #A_SHARED, #ttg.shared_memory> -> !ttg.memdesc<32x16xf16, #A_SHARED_T, #ttg.shared_memory>
  tt.return
}

// CHECK-LABEL: async_copy_global_to_local
tt.func @async_copy_global_to_local(%A : !tt.ptr<f16>, %i1 : i1) {
  %index = arith.constant 0 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : i1 -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %subview = ttg.memdesc_subview %alloc[%index, %index, %index] : !ttg.memdesc<1x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.async_copy_global_to_local %a_ptr, %subview : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %4 = ttg.local_load %subview : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  tt.return
}
// If branch inserted a barrier for %cst0, but else didn't, then the barrier should be inserted in the parent region
// CHECK-LABEL: multi_blocks
tt.func @multi_blocks(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %0 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    scf.yield
  } else {
    %cst1 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %2 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// Both branches inserted a barrier for %cst0 and %cst1, then the barrier doesn't need to be inserted in the parent region
// CHECK-LABEL: multi_blocks_join_barrier
tt.func @multi_blocks_join_barrier(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %0 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %1 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    scf.yield
  }
  // CHECK-NOT: gpu.barrier
  // CHECK: tt.return
  %a_ = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// Read yielded tensor requires a barrier
// CHECK-LABEL: multi_blocks_yield
tt.func @multi_blocks_yield(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  %a = scf.if %i1 -> (!ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %0 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    %1 = ttg.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield %1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %2 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    %3 = ttg.local_alloc %2 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  }
  %a_ = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  // CHECK: ttg.local_load
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %4 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// Even though the entry block doesn't have a barrier, the successors should have barriers
// CHECK-LABEL: multi_blocks_entry_no_shared
tt.func @multi_blocks_entry_no_shared(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  %a = scf.if %i1 -> (!ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    %cst1 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    %0 = ttg.local_load %cst1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    %1 = ttg.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield %1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  } else {
    // CHECK-NOT: gpu.barrier
    // CHECK: ttg.local_alloc
    %cst1 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield %cst1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %2 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// Conservatively add a barrier as if the branch (%i1) is never taken
// CHECK-LABEL: multi_blocks_noelse
tt.func @multi_blocks_noelse(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %0 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %1 = ttg.local_load %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// Conservatively add a barrier as if the branch (%i2) is never taken
// CHECK-LABEL: multi_blocks_nested_scf
tt.func @multi_blocks_nested_scf(%i1 : i1, %i2 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  scf.if %i1 {
    scf.if %i2 {
      // CHECK: gpu.barrier
      // CHECK-NEXT: ttg.local_load
      %0 = ttg.local_load %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
      scf.yield
    }
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %1 = ttg.local_load %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %2 = ttg.local_load %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: for
tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %a0 = ttg.local_load %a_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %b0 = ttg.local_load %b_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    scf.yield %b_shared, %a_shared, %a_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  tt.return
}

// Although a_shared and b_shared are synced before entering the loop,
// they are reassociated with aliases (c_shared) and thus require a barrier.
// CHECK-LABEL: for_alias
tt.func @for_alias(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %a0 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %b0 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %0 = ttg.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_load
    %a1 = ttg.local_load %a_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %b1 = ttg.local_load %b_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    scf.yield %c_shared, %a_shared, %b_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %r = ttg.local_load %0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

// Although cst2 is not an argument of scf.yield, its memory is reused by cst1.
// So we need a barrier both before and after cst1
// CHECK-LABEL: for_reuse
tt.func @for_reuse(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %a0 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %b0 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %0 = ttg.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    %a1 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %b1 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %1 = ttg.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    %a2 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %b2 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %2 = ttg.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    scf.yield %c_shared, %a_shared, %b_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %r = ttg.local_load %0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: for_reuse_nested
tt.func @for_reuse_nested(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %a0 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %b0 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  %0 = ttg.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    %a1 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %b1 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    %1 = ttg.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    %a_shared_next, %b_shared_next, %c_shared_next = scf.for %ivv = %lb to %ub step %step iter_args(%a_shared_nested = %a_shared_init, %b_shared_nested = %b_shared_init, %c_shared_nested = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
      // CHECK: gpu.barrier
      // CHECK-NEXT:  ttg.local_alloc
      %a2 = ttg.local_load %a_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
      %b2 = ttg.local_load %b_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
      %2 = ttg.local_alloc %a2 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
      scf.yield %c_shared_nested, %a_shared_nested, %b_shared_nested : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    }
    scf.yield %c_shared, %a_shared, %b_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT:  ttg.local_load
  %r = ttg.local_load %0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

// repeatedly write to the same shared memory addresses
// CHECK-LABEL: for_for_if
tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
      %c_shared_next_next = scf.if %i1 -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> {
        // CHECK: gpu.barrier
        // CHECK-NEXT: ttg.local_alloc
        %cst0 = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
        scf.yield %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
      } else {
        // CHECK: gpu.barrier
        // CHECK-NEXT: ttg.local_alloc
        %cst0 = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
        scf.yield %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
      }
      scf.yield %c_shared_next_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  tt.return
}

// c_block_next can either be converted from c_shared_init or c_shared_next_next
// CHECK-LABEL: for_if_for
tt.func @for_if_for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %b_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %c_shared_init = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  // CHECK: gpu.barrier
  %c_blocked = ttg.local_load %c_shared_init : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>

  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
    %c_shared_next_next = scf.if %i1 -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> {
      // CHECK: gpu.barrier
      // CHECK-NEXT: ttg.local_alloc
      %cst0 = ttg.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
      scf.yield %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    } else {
      %c_shared_ = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>) {
        // CHECK: gpu.barrier
        // CHECK-NEXT: ttg.local_load
        %c_blocked_next = ttg.local_load %c_shared_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
        scf.yield %c_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
      }
      scf.yield %c_shared_ : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
    }
    // CHECK-NOT: gpu.barrier
    %b_blocked_next = ttg.local_load %b_shared: !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
    scf.yield %a_shared, %b_shared, %c_shared_next_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  }
  tt.return
}

// CHECK-LABEL: cf_if
tt.func @cf_if(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %0 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %1 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: cf_if_else
tt.func @cf_if_else(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %0 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  %1 = ttg.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  cf.br ^bb3(%1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>)
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %2 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  %3 = ttg.local_alloc %2 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  cf.br ^bb3(%3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>)
^bb3(%arg: !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  // CHECK: ttg.local_load
  %4 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %5 = ttg.local_load %arg : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: cf_if_else_return
tt.func @cf_if_else_return(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  %b = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %0 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  %1 = ttg.local_load %b : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %2 = ttg.local_load %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  %3 = ttg.local_load %b : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: atomic_scalar
tt.func @atomic_scalar(%arg3: !tt.ptr<i32>) -> i32 {
  // CHECK-NOT: gpu.barrier
  %c0_i32 = arith.constant 0 : i32
  %1 = arith.constant dense<1.0> : tensor<128x32xf16, #AL>
  %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %4 = tt.atomic_cas acq_rel, gpu, %arg3, %c0_i32, %c0_i32 : (!tt.ptr<i32>, i32, i32) -> i32
  %3 = ttg.local_load %2 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return %4 : i32
}

// CHECK-LABEL: atomic_scalar_no_use
tt.func @atomic_scalar_no_use(%arg3: !tt.ptr<i32>) {
  %c0_i32 = arith.constant 0 : i32
  %1 = arith.constant dense<1.0> : tensor<128x32xf16, #AL>
  %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory>
  %4 = tt.atomic_cas acq_rel, gpu, %arg3, %c0_i32, %c0_i32 : (!tt.ptr<i32>, i32, i32) -> i32
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %3 = ttg.local_load %2 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory> -> tensor<128x32xf16, #AL>
  tt.return
}

}

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK-LABEL: convert_layout1
tt.func @convert_layout1(%A : !tt.ptr<f16>) {
  // CHECK-NOT: gpu.barrier
  %0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: convert_layout2
tt.func @convert_layout2(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_load
  // CHECK-NEXT: gpu.barrier
  // CHECK: ttg.local_load
  %3 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  %4 = ttg.local_load %1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: convert_layout3
tt.func @convert_layout3(%cond : i1) {
  scf.if %cond {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #A_SHARED, #ttg.shared_memory, mutable>
    // CHECK: ttg.local_load
    // CHECK-NOT: gpu.barrier
    %1 = ttg.local_load %0 : !ttg.memdesc<16x64xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x64xf16, #AL>
  } else {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    // CHECK: ttg.local_load
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttg.local_alloc
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    %2 = ttg.local_alloc %1 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  }
  tt.return
}

// CHEKC-LABEL: convert_layout4
tt.func @convert_layout4(%A : !tt.ptr<f16>, %cond : i1) {
  // CHECK-NOT: gpu.barrier
  scf.if %cond {
    tt.call @convert_layout3(%cond) : (i1) -> ()
  } else {
    tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: convert_layout5
tt.func @convert_layout5(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %0 = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_load
  // CHECK-NEXT: gpu.barrier
  // CHECK: ttg.local_load
  %3 = ttg.local_load %0 : !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  %4 = ttg.local_load %1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: single_call_sync
tt.func @single_call_sync(%A : !tt.ptr<f16>) {
  %0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK: tt.call
  // CHECK-NEXT: gpu.barrier
  tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  %1 = ttg.convert_layout %0 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #BL>
  tt.return
}

// CHECK-LABEL: single_call_no_sync
// %1 can reuse %0 in convert_layout2, which has been synced
tt.func @single_call_no_sync(%A : !tt.ptr<f16>) {
  // CHECK-NOT: gpu.barrier
  %0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  tt.call @convert_layout5(%A) : (!tt.ptr<f16>) -> ()
  %1 = ttg.convert_layout %0 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #BL>
  tt.return
}

// CHECK-LABEL: multiple_calls
tt.func @multiple_calls(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  tt.return
}

// CHECK-LABEL: if_else_calls
tt.func @if_else_calls(%A : !tt.ptr<f16>, %cond : i1) {
  scf.if %cond {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
    %cst_ = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
    %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.call
    // CHECK-NEXT: gpu.barrier
    tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
    %cst1 = ttg.local_alloc %cst_ : (tensor<16x32xf16, #AL>) -> !ttg.memdesc<16x32xf16, #A_SHARED, #ttg.shared_memory>
  } else {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
    // CHECK: tt.call
    // CHECK-NOT: gpu.barrier
    tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: for_calls
tt.func @for_calls(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.call
    tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: call_graph_1
tt.func @call_graph_1(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.call
  tt.call @convert_layout3(%cond) : (i1) -> ()
  tt.return
}

// CHECK-LABEL: call_graph_2
tt.func @call_graph_2(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  tt.call @convert_layout4(%A, %cond) : (!tt.ptr<f16>, i1) -> ()
  // CHECK: tt.call
  // CHECK-NEXT: gpu.barrier
  %cst0 = ttg.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
  tt.func public @kernel(%arg3: !tt.ptr<i32>, %arg4: !tt.ptr<f16>, %arg12: tensor<32x128xf16, #blocked>, %arg13: tensor<32x128xf32, #blocked>, %arg14: tensor<32x32xf16, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %37 = ttg.local_alloc %arg14 {allocation.offset = 0 : i32} : (tensor<32x32xf16, #blocked1>) -> !ttg.memdesc<32x32xf16, #shared, #ttg.shared_memory>
    %58 = ttg.local_alloc %arg12 : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %59 = tt.atomic_cas acq_rel, gpu, %arg3, %c0_i32, %c0_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    %60 = arith.cmpi eq, %59, %c0_i32 : i32
    cf.cond_br %60, ^bb1, ^bb2
  ^bb2:  // pred: ^bb1
    %72 = ttg.convert_layout %arg13 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #mma>
    %73 = ttg.local_load %37 : !ttg.memdesc<32x32xf16, #shared, #ttg.shared_memory> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %74 = ttg.local_load %58 : !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %75 = tt.dot %73, %74, %72, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x128xf32, #mma>
    %76 = ttg.convert_layout %75 {allocation.offset = 0 : i32} : tensor<32x128xf32, #mma> -> tensor<32x128xf32, #blocked>
    %77 = arith.truncf %76 : tensor<32x128xf32, #blocked> to tensor<32x128xf16, #blocked>
    %78 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.store %78, %77 : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// CHECK-LABEL: tma_special_cases
tt.func @tma_special_cases(%arg1: !tt.ptr<i8, 0>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %cx = arith.constant dense<1> : tensor<32xi32>
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  //      CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.init_barrier
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.ptr<i8, 0>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.wait_barrier
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.ptr<i8, 0>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>

  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.ptr<i8, 0>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttng.async_tma_gather
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_gather %arg1[%cx, %c0] %alloc, %barrier, %true : !tt.ptr<i8, 0>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, i1
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.inval_barrier
  // CHECK-NEXT: ttng.inval_barrier
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// CHECK-LABEL: tma_special_cases_cf
tt.func @tma_special_cases_cf(%arg1: !tt.ptr<i8, 0>, %i1 : i1, %arg2: tensor<256x64xf16, #blocked>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  // CF: cf.cond_br
  // SCF: scf.if
  scf.if %i1 {
    //  CHECK-NOT: gpu.barrier
    //      CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NEXT: ttng.barrier_expect
    // CHECK-NEXT: ttng.wait_barrier
    // CF-NEXT: cf.br
    // SCF-NEXT: } else {
    ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.ptr<i8, 0>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
    ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  } else {
    //  CHECK-NOT: gpu.barrier
    //      CHECK: ttg.local_store
    // CF-NEXT: cf.br
    // SCF-NEXT: }
    ttg.local_store %arg2, %alloc : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  }
  //      CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>
  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

#layout = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @warp_specialize_isolated_regions
tt.func @warp_specialize_isolated_regions(%arg0: tensor<1xi64>) {
  // CHECK-NEXT: local_alloc
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: local_store
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: barrier
  // CHECK-NEXT: local_load
  ttg.local_load %0 : !ttg.memdesc<1xi64, #layout, #smem, mutable> -> tensor<1xi64>

  // CHECK-NEXT: warp_specialize
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0() num_warps(4) {
    %cst = arith.constant dense<0> : tensor<1xi64>
    // CHECK: local_alloc
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
    // CHECK-NEXT: local_store
    ttg.local_store %cst, %1 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
    // CHECK-NEXT: barrier
    // CHECK-NEXT: local_load
    ttg.local_load %1 : !ttg.memdesc<1xi64, #layout, #smem, mutable> -> tensor<1xi64>
    // CHECK-NEXT: warp_return
    ttg.warp_return
  } : () -> ()

  tt.return
}

// CHECK-LABEL: @warp_specialize_into_default
tt.func @warp_specialize_into_default(%arg0: tensor<1xi64>) {
  // CHECK-NEXT: local_alloc
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: local_store
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: warp_specialize
  ttg.warp_specialize()
  // CHECK-NEXT: default
  default {
    // CHECK-NEXT: barrier
    // CHECK-NEXT: local_load
    ttg.local_load %0 : !ttg.memdesc<1xi64, #layout, #smem, mutable> -> tensor<1xi64>
    // CHECK-NEXT: barrier
    gpu.barrier
    // CHECK-NEXT: warp_yield
    ttg.warp_yield
  // CHECK-NEXT: () -> ()
  } : () -> ()
  // CHECK-NEXT: local_store
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  tt.return
}

// CHECK-LABEL: @default_region_cfg
tt.func @default_region_cfg(%arg0: tensor<1xi64>, %arg1: i1) {
  // CHECK-NEXT: local_alloc
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: local_store
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  // CHECK-NEXT: warp_specialize
  ttg.warp_specialize()
  // CHECK-NEXT: default
  default {
    // CHECK-NEXT: barrier
    // CHECK-NEXT: local_load
    ttg.local_load %0 : !ttg.memdesc<1xi64, #layout, #smem, mutable> -> tensor<1xi64>
    cf.cond_br %arg1, ^bb1, ^bb2
  // CHECK: ^bb1:
  ^bb1:
    // CHECK-NEXT: barrier
    gpu.barrier
    cf.br ^bb3
  ^bb2:
    cf.br ^bb3
  // CHECK: ^bb3:
  ^bb3:
    // CHECK-NEXT: warp_yield
    ttg.warp_yield
  // CHECK-NEXT: () -> ()
  } : () -> ()
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: local_store
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @direct_backedge_within_loop
tt.func @direct_backedge_within_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>, %arg5: i1) {
  // CHECK-NEXT: constant
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
  // CHECK-NEXT: local_alloc
  %0 = ttg.local_alloc %cst : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
  // CHECK-NEXT: barrier
  // CHECK-NEXT: local_load
  %1 = ttg.local_load %0 : !ttg.memdesc<128x32xf16, #shared, #smem> -> tensor<128x32xf16, #blocked>
  // CHECK-NEXT: br
  cf.br ^bb1(%arg0, %0 : index, !ttg.memdesc<128x32xf16, #shared, #smem>)
^bb1(%2: index, %3: !ttg.memdesc<128x32xf16, #shared, #smem>):
  cf.cond_br %arg5, ^bb2, ^bb3
// CHECK: ^bb2:
^bb2:
  // CHECK-NEXT: barrier
  // CHECK-NEXT: local_alloc
  %4 = ttg.local_alloc %cst : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
  // CHECK-NEXT: br
  cf.br ^bb1(%arg1, %4 : index, !ttg.memdesc<128x32xf16, #shared, #smem>)
// CHECK: ^bb3
^bb3:
  // CHECK-NEXT: barrier
  // CHECK-NEXT: local_load
  %5 = ttg.local_load %3 : !ttg.memdesc<128x32xf16, #shared, #smem> -> tensor<128x32xf16, #blocked>
  // CHECK-NEXT: cond_br
  cf.cond_br %arg5, ^bb3, ^bb4
^bb4:
  tt.return
}

}

// -----

// CHECK-LABEL: tmem_copy_after_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @tmem_copy_after_alloc(%arg0: tensor<1x2048xf8E4M3FN, #blocked>) {
    // CHECK: local_alloc
    %0 = ttg.local_alloc %arg0 {allocation.offset = 53248 : i32} : (tensor<1x2048xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x2048xf8E4M3FN, #shared, #smem>
    // CHECK: tmem_alloc
    %1 = ttng.tmem_alloc  {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // gpu.barrier
    // CHECK: tmem_copy
    ttng.tmem_copy %0, %1,  : (!ttg.memdesc<1x2048xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
