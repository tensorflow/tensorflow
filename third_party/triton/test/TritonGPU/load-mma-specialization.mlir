// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-load-mma-specialization -canonicalize -cse | FileCheck %s

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-DAG: [[SHARED:#.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-DAG: [[ACC_TMEM:#.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK: @warp_specialize_tma_matmul
// CHECK-SAME: [[K_TILES:%arg[0-9]+]]
// CHECK-SAME: [[OFF_M:%arg[0-9]+]]
// CHECK-SAME: [[OFF_N:%arg[0-9]+]]
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @warp_specialize_tma_matmul(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>
) {
  // CHECK-DAG: [[TRUE:%.*]] = arith.constant true
  %true = arith.constant true
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-DAG: [[BLOCK_K:%.*]] = arith.constant 64 : i32
  %BLOCK_K = arith.constant 64 : i32
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  // CHECK-DAG: [[CNEG1:%.*]] = arith.constant -1 : i32
  // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : i32

  // CHECK:      [[C_TMEM:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, [[ACC_TMEM]], #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[MMA_MBAR_VIEW:%.*]] = ttg.memdesc_subview [[MMA_MBAR]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR_VIEW]], 1
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[C_TMEM]]

  // CHECK-NEXT: [[A_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]
  // CHECK-NEXT: [[B_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]

  // CHECK-NEXT: [[READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[READY_MBAR0:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: [[READY_MBAR1:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[C1]]]
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR1]], 1
  // CHECK-NEXT: [[OPER_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[OPER_MBAR0:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR0]], 1
  // CHECK-NEXT: [[OPER_MBAR1:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[C1]]]
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR1]], 1

  // CHECK-NEXT: {{[0-9]+}}:3 = scf.for [[K:%arg[0-9]+]] = [[C0]] to [[K_TILES]] step [[C1]]
  // CHECK-SAME: [[OPER_IDX:%arg[0-9]+]] = [[CNEG1]]
  // CHECK-SAME: [[OPER_PHASE:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: -> (i32, i32, i32)
  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[OPER_IDX_NEXT:%.*]] = arith.addi [[OPER_IDX]], [[C1]]
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[OPER_IDX_NEXT]], [[C2]]
    // CHECK-NEXT: [[IDX:%.*]] = arith.select [[ROLLOVER]], [[C0]], [[OPER_IDX_NEXT]]
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[OPER_PHASE]], [[C1]]
    // CHECK-NEXT: [[PHASE:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[OPER_PHASE]]

    // CHECK-NEXT: [[NEXT_MMA_PHASE:%.*]] = arith.xori [[MMA_PHASE]], [[C1]]

    // CHECK-NEXT: [[OFF_K:%.*]] = arith.muli [[K]], [[BLOCK_K]]
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK-NEXT: [[READY_MBAR:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[IDX]]]
    // CHECK-NEXT: ttng.wait_barrier [[READY_MBAR]], [[PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[OPER_MBAR:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[IDX]]]
    // CHECK-NEXT: ttng.barrier_expect [[OPER_MBAR]], 32768 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[A_BUF:%.*]] = ttg.memdesc_subview [[A_BUFS]][[[IDX]], [[C0]], [[C0]]]
    // CHECK-NEXT: [[A_DESC_PTR:%.*]] = ttng.tensor_desc_to_tma_ptr [[A_DESC]]
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[A_DESC_PTR]][[[OFF_M]], [[OFF_K]]] [[A_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 0 : i32}
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    // CHECK-NEXT: [[B_BUF:%.*]] = ttg.memdesc_subview [[B_BUFS]][[[IDX]], [[C0]], [[C0]]]
    // CHECK-NEXT: [[B_DESC_PTR:%.*]] = ttng.tensor_desc_to_tma_ptr [[B_DESC]]
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[B_DESC_PTR]][[[OFF_N]], [[OFF_K]]] [[B_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 0 : i32}
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK-NEXT: ttng.wait_barrier [[OPER_MBAR]], [[PHASE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[B_T:%.*]] = ttg.memdesc_trans [[B_BUF]] {order = array<i32: 1, 0>, ttg.partition = 1 : i32}
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: ttng.tc_gen5_mma [[A_BUF]], [[B_T]], [[C_TMEM]], [[TRUE]], [[TRUE]], [[MMA_MBAR]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR]], 1 {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()

    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: yield [[IDX]], [[PHASE]], [[NEXT_MMA_PHASE]]
    scf.yield %c : tensor<128x128xf32, #acc_layout>

  // CHECK-NEXT: {tt.warp_specialize, ttg.partition.stages = [0 : i32, 2 : i32]}
  } {tt.warp_specialize}

  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[OPER_MBARS]]

  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[READY_MBARS]]

  // CHECK-NEXT: ttg.local_dealloc [[B_BUFS]]
  // CHECK-NEXT: ttg.local_dealloc [[A_BUFS]]

  // CHECK-NEXT: [[RESULT:%.*]] = ttng.tmem_load [[C_TMEM]]
  // CHECK-NEXT: ttng.inval_barrier [[MMA_MBAR_VIEW]]
  // CHECK-NEXT: ttg.local_dealloc [[MMA_MBAR]]

  // CHECK-NEXT: "use"([[RESULT]])
  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

}
