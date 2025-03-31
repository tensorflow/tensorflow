// RUN: triton-opt %s -tritongpu-tc05mma-pipeline=disable-expander=true -canonicalize | FileCheck --dump-input-context=50 %s --check-prefix=CHECK-LOWER
// RUN: triton-opt %s -tritongpu-tc05mma-pipeline -canonicalize | FileCheck --dump-input-context=50 %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 =  #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_scales_encoding<>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @chained_dot_no_multibuf_acc
  // CHECK-LOWER-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32
  // CHECK-LOWER: ttng.tmem_store %[[C0_F]], %[[TMEM_BUF]]
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_BUF]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]
  // CHECK-LOWER: ttng.tmem_load %[[TMEM_BUF]]

  // CHECK-LABEL: @chained_dot_no_multibuf_acc
  tt.func public @chained_dot_no_multibuf_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @chained_dot_wait_before_store
  // CHECK-LOWER-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32
  // CHECK-LOWER: ttng.tmem_store %[[C0_F]], %[[TMEM_BUF]]
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_BUF]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   scf.if
  // CHECK-LOWER:     ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]]
  // CHECK-LOWER:     %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_BUF]]
  // CHECK-LOWER:     tt.store %{{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]
  // CHECK-LOWER: ttng.tmem_load %[[TMEM_BUF]]

  // CHECK-LABEL: @chained_dot_wait_before_store
  tt.func public @chained_dot_wait_before_store(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %cnd: i1) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Verify that we still can pipeline the mma when the subview is in the previous iteration.
  // CHECK-LOWER-LABEL: @subview_dist_1
  // CHECK-LOWER: ttng.tmem_alloc
  // CHECK-LOWER: ttng.tmem_store
  // CHECK-LOWER: scf.for
  // CHECK-LOWER:   ttng.tc_gen5_mma
  // CHECK-LOWER:   scf.yield
  // CHECK-LOWER: ttng.tmem_load

  // CHECK-LABEL: @subview_dist_1
  tt.func public @subview_dist_1(%arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %A_sh0 = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res, %_ = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %A_sh_arg = %A_sh0) -> (tensor<128x128xf32, #blocked>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>)  : i32 {
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh_arg, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %A_sh1 = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      scf.yield %acc_res, %A_sh1 : tensor<128x128xf32, #blocked>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @multibuf_acc
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]], %[[ACC_INS_IDX:.+]] = %[[C0]], %[[ACC_EXT_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[ACC_INS_IDX_P1:.+]] = arith.addi %[[ACC_INS_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_INS_WRAP:.+]] = arith.cmpi eq, %[[ACC_INS_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT:.+]] = arith.select %[[ACC_INS_WRAP]], %[[C0]], %[[ACC_INS_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_EXT_IDX_P1:.+]] = arith.addi %[[ACC_EXT_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_EXT_WRAP:.+]] = arith.cmpi eq, %[[ACC_EXT_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT:.+]] = arith.select %[[ACC_EXT_WRAP]], %[[C0]], %[[ACC_EXT_IDX_P1]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT]],
  // CHECK-LOWER:   ttng.tmem_store {{.*}}, %[[TMEM_INS_SLICE]], %[[TRUE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT]],
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_INS_SLICE]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[TMEM_EXT_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_EXT_NEXT]],
  // CHECK-LOWER:   %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_EXT_SLICE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[ACC_INS_NEXT]], %[[ACC_EXT_NEXT]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @multibuf_acc
  tt.func public @multibuf_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Do not pipeline the mma, as multibuffering is disabled, and would need to wait in the
  // every iteration of the loop anyway.
  // CHECK-LOWER-LABEL: @disable_multibuf_acc
  // CHECK-LOWER-NOT: ttng.wait_barrier
  // CHECK-LABEL: @disable_multibuf_acc
  tt.func public @disable_multibuf_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.disallow_acc_multi_buffer}
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @do_not_pipeline_two_dots
  // CHECK-LOWER-NOT: triton.pipeline_stage

  // CHECK-LABEL: @do_not_pipeline_two_dots
  tt.func public @do_not_pipeline_two_dots(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc_tm2 = ttng.tmem_alloc %acc_res : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm2, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res2 = ttng.tmem_load %acc_tm2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %res_ptr, %acc_res2 : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @multibuf_acc_sel_override
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]], %[[ACC_INS_IDX:.+]] = %[[C0]], %[[ACC_EXT_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[ACC_INS_IDX_P1:.+]] = arith.addi %[[ACC_INS_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_INS_WRAP:.+]] = arith.cmpi eq, %[[ACC_INS_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT:.+]] = arith.select %[[ACC_INS_WRAP]], %[[C0]], %[[ACC_INS_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_EXT_IDX_P1:.+]] = arith.addi %[[ACC_EXT_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_EXT_WRAP:.+]] = arith.cmpi eq, %[[ACC_EXT_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT:.+]] = arith.select %[[ACC_EXT_WRAP]], %[[C0]], %[[ACC_EXT_IDX_P1]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT]],
  // CHECK-LOWER:   ttng.tmem_store {{.*}}, %[[TMEM_INS_SLICE]], %[[TRUE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT]],
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_INS_SLICE]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[TMEM_EXT_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_EXT_NEXT]],
  // CHECK-LOWER:   %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_EXT_SLICE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[ACC_INS_NEXT]], %[[ACC_EXT_NEXT]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @multibuf_acc
  tt.func public @multibuf_acc_sel_override(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      // %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>
      %acc = arith.select %cnd, %cst1, %cst0 : tensor<128x128xf32, #blocked>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @multibuf_acc_unused
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[FALSE:.+]] = arith.constant false
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]], %[[ACC_INS_IDX:.+]] = %[[C0]], %[[ACC_EXT_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[ACC_INS_IDX_P1:.+]] = arith.addi %[[ACC_INS_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_INS_WRAP:.+]] = arith.cmpi eq, %[[ACC_INS_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT:.+]] = arith.select %[[ACC_INS_WRAP]], %[[C0]], %[[ACC_INS_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_EXT_IDX_P1:.+]] = arith.addi %[[ACC_EXT_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_EXT_WRAP:.+]] = arith.cmpi eq, %[[ACC_EXT_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT:.+]] = arith.select %[[ACC_EXT_WRAP]], %[[C0]], %[[ACC_EXT_IDX_P1]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT]],
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_INS_SLICE]], %[[FALSE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[TMEM_EXT_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_EXT_NEXT]],
  // CHECK-LOWER:   %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_EXT_SLICE]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[ACC_INS_NEXT]], %[[ACC_EXT_NEXT]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @multibuf_acc_unused
  tt.func public @multibuf_acc_unused(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32) attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %false, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @acc_reinit_under_sel
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
  // CHECK-LOWER: %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[C0]]
  // CHECK-LOWER: ttng.tmem_store %[[C0_F]], %[[TMEM_INS_SLICE]]
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]], %[[ACC_INS_IDX:.+]] = %[[C0]], %[[ACC_EXT_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_IDX]],
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_INS_SLICE]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]]  {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_INS_IDX_P1:.+]] = arith.addi %[[ACC_INS_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_INS_WRAP:.+]] = arith.cmpi eq, %[[ACC_INS_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT:.+]] = arith.select %[[ACC_INS_WRAP]], %[[C0]], %[[ACC_INS_IDX_P1]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT_PRED:.+]] = arith.select %[[CND:.+]], %[[ACC_INS_NEXT]], %[[ACC_INS_IDX]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_EXT_IDX_P1:.+]] = arith.addi %[[ACC_EXT_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_EXT_WRAP:.+]] = arith.cmpi eq, %[[ACC_EXT_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT:.+]] = arith.select %[[ACC_EXT_WRAP]], %[[C0]], %[[ACC_EXT_IDX_P1]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT_PRED:.+]] = arith.select %[[CND]], %[[ACC_EXT_NEXT]], %[[ACC_EXT_IDX]] {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_NEXT_PRED]],
  // CHECK-LOWER:   ttng.tmem_store %[[C0_F]], %[[TMEM_INS_SLICE]], %[[CND]]
  // CHECK-LOWER:   scf.if %[[CND]]
  // CHECK-LOWER:     %[[TMEM_EXT_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_EXT_IDX]],
  // CHECK-LOWER:     %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_EXT_SLICE]]
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[ACC_INS_NEXT_PRED]], %[[ACC_EXT_NEXT_PRED]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @acc_reinit_under_sel
  tt.func public @acc_reinit_under_sel(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_acc = arith.select %cnd, %cst, %acc_res : tensor<128x128xf32, #blocked>
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %new_acc : tensor<128x128xf32, #blocked>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Do not pipeline if multibufferring is disallowed and we are physilcally override accumulator
  // in the loop body.
  // CHECK-LOWER-LABEL: @acc_reinit_under_sel_disallow_multibuffer
  // CHECK-LOWER:     ttng.tc_gen5_mma
  // CHECK-LOWER-NOT: ttng.wait_barrier

  // CHECK-LABEL: @acc_reinit_under_sel_disallow_multibuffer
  tt.func public @acc_reinit_under_sel_disallow_multibuffer(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_acc = arith.select %cnd, %cst, %acc_res : tensor<128x128xf32, #blocked>
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %new_acc : tensor<128x128xf32, #blocked>
    } {tt.disallow_acc_multi_buffer}
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @acc_reinit_under_if_acc_flag
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[FALSE:.+]] = arith.constant false
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
  // CHECK-LOWER: %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[C0]]
  // CHECK-LOWER: ttng.tmem_store %[[C0_F]], %[[TMEM_INS_SLICE]]
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[ACC_USE:.+]] = %[[FALSE]], %[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]], %[[ACC_INS_IDX:.+]] = %[[C0]], %[[ACC_EXT_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[TMEM_INS_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_INS_IDX]],
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_INS_SLICE]], %[[ACC_USE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]]  {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_USE_NEXT:.+]] = arith.xori %[[CND:.+]], %[[TRUE]]
  // CHECK-LOWER:   %[[ACC_INS_IDX_P1:.+]] = arith.addi %[[ACC_INS_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_INS_WRAP:.+]] = arith.cmpi eq, %[[ACC_INS_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT:.+]] = arith.select %[[ACC_INS_WRAP]], %[[C0]], %[[ACC_INS_IDX_P1]]
  // CHECK-LOWER:   %[[ACC_INS_NEXT_PRED:.+]] = arith.select %[[CND]], %[[ACC_INS_NEXT]], %[[ACC_INS_IDX]]
  // CHECK-LOWER:   %[[ACC_EXT_IDX_P1:.+]] = arith.addi %[[ACC_EXT_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[ACC_EXT_WRAP:.+]] = arith.cmpi eq, %[[ACC_EXT_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT:.+]] = arith.select %[[ACC_EXT_WRAP]], %[[C0]], %[[ACC_EXT_IDX_P1]]
  // CHECK-LOWER:   %[[ACC_EXT_NEXT_PRED:.+]] = arith.select %[[CND]], %[[ACC_EXT_NEXT]], %[[ACC_EXT_IDX]]
  // CHECK-LOWER:   scf.if %[[CND]]
  // CHECK-LOWER:     %[[TMEM_EXT_SLICE:.+]] = ttg.memdesc_subview %[[TMEM_BUF]][%[[ACC_EXT_IDX]],
  // CHECK-LOWER:     %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_EXT_SLICE]]
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   scf.yield %[[ACC_USE_NEXT]], %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[ACC_INS_NEXT_PRED]], %[[ACC_EXT_NEXT_PRED]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @acc_reinit_under_if_acc_flag
  tt.func public @acc_reinit_under_if_acc_flag(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %accUse = %false) -> (tensor<128x128xf32, #blocked>, i1)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %accUse, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_accUse = arith.select %cnd, %false, %true : i1
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %acc_res, %new_accUse : tensor<128x128xf32, #blocked>, i1
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @acc_reinit_under_if_acc_flag_disallow_multibuffer
  // CHECK-LOWER-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-LOWER-DAG: %[[FALSE:.+]] = arith.constant false
  // CHECK-LOWER-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-LOWER-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-LOWER-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-LOWER-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-LOWER: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32
  // CHECK-LOWER: ttng.tmem_store %[[C0_F]], %[[TMEM_BUF]]
  // CHECK-LOWER: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK-LOWER: scf.for {{.*}} iter_args(%[[ACC_USE:.+]] = %[[FALSE]], %[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C0]])
  // CHECK-LOWER:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]
  // CHECK-LOWER:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_BUF]], %[[ACC_USE]], %[[TRUE]], %[[BAR_SLICE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]]  {triton.pipeline_stage = 1 : i32}
  // CHECK-LOWER:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK-LOWER:   %[[BAR_WRAP:.+]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK-LOWER:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK-LOWER:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   %[[ACC_USE_NEXT:.+]] = arith.xori %[[CND:.+]], %[[TRUE]]
  // CHECK-LOWER:   scf.if %{{.*}}
  // CHECK-LOWER:     ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]]
  // CHECK-LOWER:     %[[ACC_RES:.+]] = ttng.tmem_load %[[TMEM_BUF]]
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } {triton.pipeline_stage = 0 : i32}
  // CHECK-LOWER:   scf.yield %[[ACC_USE_NEXT]], %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]]
  // CHECK-LOWER: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK-LOWER: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK-LOWER: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK-LOWER: ttg.local_dealloc %[[BAR_BUF]]

  // CHECK-LABEL: @acc_reinit_under_if_acc_flag_disallow_multibuffer
  tt.func public @acc_reinit_under_if_acc_flag_disallow_multibuffer(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %accUse = %false) -> (tensor<128x128xf32, #blocked>, i1)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %accUse, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_accUse = arith.select %cnd, %false, %true : i1
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %acc_res, %new_accUse : tensor<128x128xf32, #blocked>, i1
    } {tt.disallow_acc_multi_buffer}
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @acc_used_if_else
  // CHECK-LOWER: ttng.tmem_alloc
  // CHECK-LOWER: ttng.tmem_store
  // CHECK-LOWER: scf.for
  // CHECK-LOWER:   ttng.tc_gen5_mma
  // CHECK-LOWER:   %[[EXT_SLICE:.+]] = ttg.memdesc_subview
  // CHECK-LOWER:   %[[ACC_RES:.+]] = ttng.tmem_load %[[EXT_SLICE]]
  // CHECK-LOWER:   scf.if
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } else {
  // CHECK-LOWER:     arith.addf %[[ACC_RES]], %[[ACC_RES]]

  // CHECK-LABEL: @acc_used_if_else
  tt.func public @acc_used_if_else(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %accUse = %false) -> (tensor<128x128xf32, #blocked>, i1)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %accUse, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_accUse = arith.select %cnd, %false, %true : i1
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      } else {
        %acc_res2 = arith.addf %acc_res, %acc_res : tensor<128x128xf32, #blocked>
        tt.store %res_ptr, %acc_res2 : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %acc_res, %new_accUse : tensor<128x128xf32, #blocked>, i1
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LOWER-LABEL: @acc_used_if_and_outside
  // CHECK-LOWER: ttng.tmem_alloc
  // CHECK-LOWER: ttng.tmem_store
  // CHECK-LOWER: scf.for
  // CHECK-LOWER:   ttng.tc_gen5_mma
  // CHECK-LOWER:   %[[EXT_SLICE:.+]] = ttg.memdesc_subview
  // CHECK-LOWER:   %[[ACC_RES:.+]] = ttng.tmem_load %[[EXT_SLICE]]
  // CHECK-LOWER:   %[[ACC_RES2:.+]] = arith.addf %[[ACC_RES]], %[[ACC_RES]]
  // CHECK-LOWER:   scf.if
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES]]
  // CHECK-LOWER:   } else {
  // CHECK-LOWER:     tt.store {{.*}}, %[[ACC_RES2]]

  // CHECK-LABEL: @acc_used_if_and_outside
  tt.func public @acc_used_if_and_outside(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg3: i32, %cnd: i1) attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %accUse = %false) -> (tensor<128x128xf32, #blocked>, i1)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %accUse, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %new_accUse = arith.select %cnd, %false, %true : i1
      %acc_res2 = arith.addf %acc_res, %acc_res : tensor<128x128xf32, #blocked>
      scf.if %cnd {
        tt.store %res_ptr, %acc_res : tensor<128x128x!tt.ptr<f32>, #blocked>
      } else {
        tt.store %res_ptr, %acc_res2 : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %acc_res, %new_accUse : tensor<128x128xf32, #blocked>, i1
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipeline_tc05mma
  // CHECK: (%[[UB_ARG:[0-9a-z]+]]: i32,
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[A_SH:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16
  // CHECK: %[[B_SH:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16

  // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc
  // CHECK: ttng.tmem_store %[[CST]], %[[TMEM]], %[[TRUE]]

  // Barrier allocation:
  // CHECK: %[[BAR_SH:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: %[[BAR_SLICE0:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[C0]]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK: %[[BAR_SLICE1:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[C1]]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE1]], 1

  // Peeled prologue:
  // CHECK-DAG: %[[I0_PRED:.*]] = arith.cmpi sgt, %[[UB_ARG]], %[[C0]] : i32
  // CHECK-DAG: %[[A_SLICE0:.*]] = ttg.memdesc_subview %[[A_SH]][%[[C0]], %[[C0]], %[[C0]]]
  // CHECK-DAG: %[[B_SLICE0:.*]] = ttg.memdesc_subview %[[B_SH]][%[[C0]], %[[C0]], %[[C0]]]
  // CHECK-DAG: %[[BAR_SLICE0_1:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[C0]]]
  // CHECK: ttng.tc_gen5_mma %[[A_SLICE0]], %[[B_SLICE0]], %[[TMEM]], %[[TRUE]], %[[I0_PRED]], %[[BAR_SLICE0_1]]

  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[UB_ARG]] step %[[C1]] iter_args({{.*}} %[[PHASE:.[^,]+]] = %[[C0]], %[[BAR_IDX:[^,]+]] = %[[C1]], %[[BAR_SLICE_PREV:.[^,]+]] = %[[BAR_SLICE0_1]], %[[PHASE_PREV:.[^,]+]] = %[[C0]]
  // CHECK:   %[[UB_M1:.*]] = arith.subi %[[UB_ARG]], %[[C1]]
  // CHECK:   %[[IN_PRED:.*]] = arith.cmpi slt, %[[IV]], %[[UB_M1]]
  // CHECK:   %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[BAR_IDX]]]
  // CHECK:   ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM]], %[[TRUE]], %[[IN_PRED]], %[[BAR_SLICE]]
  // CHECK:   ttng.wait_barrier %[[BAR_SLICE_PREV]], %[[PHASE_PREV]]

  // CHECK:   %[[BAR_IDX_P1:.*]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK:   %[[BAR_IDX_WRAP:.*]] = arith.cmpi eq, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK:   %[[BAR_IDX_NEXT:.*]] = arith.select %[[BAR_IDX_WRAP]], %[[C0]], %[[BAR_IDX_P1]]

  // CHECK:   %[[XOR:.*]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK:   %[[NEXT_PHASE:.*]] = arith.select %[[BAR_IDX_WRAP]], %[[XOR]], %[[PHASE]]
  // CHECK:   scf.yield {{.*}}, %[[NEXT_PHASE]], %[[BAR_IDX_NEXT]], %[[BAR_SLICE]], %[[PHASE]]
  // CHECK: %[[BAR_SLICE0:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[C0]]]
  // CHECK: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK: %[[BAR_SLICE1:.*]] = ttg.memdesc_subview %[[BAR_SH]][%[[C1]]]
  // CHECK: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK: ttg.local_dealloc %[[BAR_SH]]

  tt.func public @pipeline_tc05mma(%arg0: i32, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked1> attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %3 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    %4 = ttg.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %5 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %6 = ttg.async_copy_global_to_local %arg1, %4 mask %5 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %7 = ttg.async_commit_group %6
    %8 = ttg.memdesc_subview %2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %9 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %10 = ttg.async_copy_global_to_local %arg2, %8 mask %9 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %11 = ttg.async_commit_group %10
    %12:5 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %c0_i32, %arg5 = %c-1_i32, %arg6 = %7, %arg7 = %11, %acc = %cst) -> (i32, i32, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #blocked1>)  : i32 {
      %15 = arith.subi %arg0, %c1_i32 : i32
      %16 = arith.cmpi slt, %arg3, %15 : i32
      %17 = arith.addi %arg5, %c1_i32 : i32
      %18 = arith.cmpi slt, %17, %c2_i32 : i32
      %19 = arith.select %18, %17, %c0_i32 : i32
      %20 = ttg.memdesc_subview %1[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %21 = ttg.async_wait %arg7 {num = 0 : i32}
      %22 = ttg.memdesc_subview %2[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %20, %22, %tmem, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %tmem : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %23 = arith.addi %arg4, %c1_i32 : i32
      %24 = arith.cmpi slt, %23, %c2_i32 : i32
      %25 = arith.select %24, %23, %c0_i32 : i32
      %26 = ttg.memdesc_subview %1[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %27 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %28 = ttg.async_copy_global_to_local %arg1, %26 mask %27 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %29 = ttg.async_commit_group %28
      %30 = ttg.memdesc_subview %2[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %31 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %32 = ttg.async_copy_global_to_local %arg2, %30 mask %31 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %33 = ttg.async_commit_group %32
      scf.yield %25, %19, %29, %33, %acc_res : i32, i32, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #blocked1>
    }
    %13 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %1 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return %12#4 : tensor<128x128xf32, #blocked1>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipeline_tc05mma_scaled
  // CHECK:   ttng.tc_gen5_mma_scaled
  // MMA pipeline should not apply since scales are not passed in shmem
  // CHECK-NOT:   ttng.wait_barrier

  tt.func public @pipeline_tc05mma_scaled(%arg0: i32, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %scale_A: tensor<128x4x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %scale_B: tensor<128x4x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked1> attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %scale_A_SMEM = ttg.local_alloc : () -> !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %scale_B_SMEM = ttg.local_alloc : () -> !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %3 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    %4 = ttg.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %5 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %6 = ttg.async_copy_global_to_local %arg1, %4 mask %5 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %7 = ttg.async_commit_group %6
    %8 = ttg.memdesc_subview %2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %9 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %10 = ttg.async_copy_global_to_local %arg2, %8 mask %9 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %11 = ttg.async_commit_group %10
    %43 = ttg.memdesc_subview %scale_A_SMEM[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %44 = tt.splat %3 : i1 -> tensor<128x4xi1, #blocked>
    %45 = ttg.async_copy_global_to_local %scale_A, %43 mask %44 : tensor<128x4x!tt.ptr<i8>, #blocked> -> <128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %46 = ttg.async_commit_group %45
    %47 = ttg.memdesc_subview %scale_B_SMEM[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %48 = tt.splat %3 : i1 -> tensor<128x4xi1, #blocked>
    %49 = ttg.async_copy_global_to_local %scale_B, %47 mask %48 : tensor<128x4x!tt.ptr<i8>, #blocked> -> <128x4xi8, #shared1, #ttg.shared_memory, mutable>
    %50 = ttg.async_commit_group %49
    %51 = ttg.async_wait %50 {num = 0 : i32}
    %12:9 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %c0_i32, %arg5 = %c-1_i32, %arg6 = %7, %arg7 = %11, %arg8 = %43, %arg8_token = %51, %arg9 = %47, %arg9_token = %51, %acc = %cst) -> (i32, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, tensor<128x128xf32, #blocked1>)  : i32 {
      %15 = arith.subi %arg0, %c1_i32 : i32
      %16 = arith.cmpi slt, %arg3, %15 : i32
      %17 = arith.addi %arg5, %c1_i32 : i32
      %18 = arith.cmpi slt, %17, %c2_i32 : i32
      %19 = arith.select %18, %17, %c0_i32 : i32
      %20 = ttg.memdesc_subview %1[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %21 = ttg.async_wait %arg7 {num = 0 : i32}
      %22 = ttg.memdesc_subview %2[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %122 = ttg.local_load %arg8 token %arg8_token : !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable> -> tensor<128x4xi8, #blocked2>
      %123 = ttg.local_load %arg9 token %arg9_token : !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable> -> tensor<128x4xi8, #blocked2>
      %125 = ttg.convert_layout %122 : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #blocked3>
      %126 = ttg.convert_layout %123 : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #blocked3>
      %127 = ttng.tmem_alloc %125 : (tensor<128x4xi8, #blocked3>) -> !ttg.memdesc<128x4xi8, #tmem1, #ttng.tensor_memory>
      %128 = ttng.tmem_alloc %126 : (tensor<128x4xi8, #blocked3>) -> !ttg.memdesc<128x4xi8, #tmem1, #ttng.tensor_memory>
      %tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma_scaled %20, %22, %tmem, %127, %128, %true, %true lhs = e5m2 rhs = e5m2: (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x4xi8, #tmem1, #ttng.tensor_memory>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %tmem : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %23 = arith.addi %arg4, %c1_i32 : i32
      %24 = arith.cmpi slt, %23, %c2_i32 : i32
      %25 = arith.select %24, %23, %c0_i32 : i32
      %26 = ttg.memdesc_subview %1[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %27 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %28 = ttg.async_copy_global_to_local %arg1, %26 mask %27 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %29 = ttg.async_commit_group %28
      %30 = ttg.memdesc_subview %2[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %31 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %32 = ttg.async_copy_global_to_local %arg2, %30 mask %31 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %33 = ttg.async_commit_group %32
      %34 = ttg.memdesc_subview %scale_A_SMEM[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>
      %35 = tt.splat %16 : i1 -> tensor<128x4xi1, #blocked>
      %36 = ttg.async_copy_global_to_local %scale_A, %34 mask %35 : tensor<128x4x!tt.ptr<i8>, #blocked> -> <128x4xi8, #shared1, #ttg.shared_memory, mutable>
      %37 = ttg.async_commit_group %36
      %38 = ttg.memdesc_subview %scale_B_SMEM[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>
      %39 = tt.splat %16 : i1 -> tensor<128x4xi1, #blocked>
      %40 = ttg.async_copy_global_to_local %scale_B, %38 mask %39 : tensor<128x4x!tt.ptr<i8>, #blocked> -> <128x4xi8, #shared1, #ttg.shared_memory, mutable>
      %41 = ttg.async_commit_group %40
      %42 = ttg.async_wait %41 {num = 0 : i32}
      scf.yield %25, %19, %29, %33, %34, %42, %38, %42, %acc_res : i32, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, tensor<128x128xf32, #blocked1>
    }
    %13 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %1 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %scale_A_SMEM : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable>
    ttg.local_dealloc %scale_B_SMEM : !ttg.memdesc<2x128x4xi8, #shared1, #ttg.shared_memory, mutable>
    tt.return %12#8 : tensor<128x128xf32, #blocked1>
  }
}


module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipeline_tc05mma_scaled_shmem
  // CHECK:   ttng.wait_barrier

  tt.func public @pipeline_tc05mma_scaled_shmem(%arg0: i32, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %scale_A: tensor<1x512x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %scale_B: tensor<1x512x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked1> attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_1 = arith.constant dense<127> : tensor<128x4xi8, #blocked4>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %scale_A_SMEM = ttg.local_alloc : () -> !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %scale_B_SMEM = ttg.local_alloc : () -> !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %3 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    %4 = ttg.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %5 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %6 = ttg.async_copy_global_to_local %arg1, %4 mask %5 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %7 = ttg.async_commit_group %6
    %8 = ttg.memdesc_subview %2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %9 = tt.splat %3 : i1 -> tensor<128x128xi1, #blocked>
    %10 = ttg.async_copy_global_to_local %arg2, %8 mask %9 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
    %11 = ttg.async_commit_group %10

    %43 = ttg.memdesc_subview %scale_A_SMEM[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %44 = tt.splat %3 : i1 -> tensor<1x512xi1, #blocked>
    %45 = ttg.async_copy_global_to_local %scale_A, %43 mask %44 : tensor<1x512x!tt.ptr<i8>, #blocked> -> <1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %46 = ttg.async_commit_group %45
    %47 = ttg.memdesc_subview %scale_B_SMEM[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %48 = tt.splat %3 : i1 -> tensor<1x512xi1, #blocked>
    %49 = ttg.async_copy_global_to_local %scale_B, %47 mask %48 : tensor<1x512x!tt.ptr<i8>, #blocked> -> <1x512xi8, #shared1, #ttg.shared_memory, mutable>
    %50 = ttg.async_commit_group %49
    %51 = ttg.async_wait %50 {num = 0 : i32}
    %12:9 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %c0_i32, %arg5 = %c-1_i32, %arg6 = %7, %arg7 = %11, %arg8 = %43, %arg8_token = %51, %arg9 = %47, %arg9_token = %51, %acc = %cst) -> (i32, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, tensor<128x128xf32, #blocked1>)  : i32 {
      %15 = arith.subi %arg0, %c1_i32 : i32
      %16 = arith.cmpi slt, %arg3, %15 : i32
      %17 = arith.addi %arg5, %c1_i32 : i32
      %18 = arith.cmpi slt, %17, %c2_i32 : i32
      %19 = arith.select %18, %17, %c0_i32 : i32
      %20 = ttg.memdesc_subview %1[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %21 = ttg.async_wait %arg7 {num = 0 : i32}
      %22 = ttg.memdesc_subview %2[%19, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>

      %tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cst_scale = ttng.tmem_alloc %cst_1 : (tensor<128x4xi8, #blocked4>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %20, %22, %tmem, %arg8, %cst_scale, %true, %true lhs = e5m2 rhs = e5m2: (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %tmem : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %23 = arith.addi %arg4, %c1_i32 : i32
      %24 = arith.cmpi slt, %23, %c2_i32 : i32
      %25 = arith.select %24, %23, %c0_i32 : i32
      %26 = ttg.memdesc_subview %1[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %27 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %28 = ttg.async_copy_global_to_local %arg1, %26 mask %27 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %29 = ttg.async_commit_group %28
      %30 = ttg.memdesc_subview %2[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %31 = tt.splat %16 : i1 -> tensor<128x128xi1, #blocked>
      %32 = ttg.async_copy_global_to_local %arg2, %30 mask %31 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #ttg.shared_memory, mutable>
      %33 = ttg.async_commit_group %32
      %34 = ttg.memdesc_subview %scale_A_SMEM[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>
      %35 = tt.splat %16 : i1 -> tensor<1x512xi1, #blocked>
      %36 = ttg.async_copy_global_to_local %scale_A, %34 mask %35 : tensor<1x512x!tt.ptr<i8>, #blocked> -> <1x512xi8, #shared1, #ttg.shared_memory, mutable>
      %37 = ttg.async_commit_group %36
      %38 = ttg.memdesc_subview %scale_B_SMEM[%25, %c0_i32, %c0_i32] : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>
      %39 = tt.splat %16 : i1 -> tensor<1x512xi1, #blocked>
      %40 = ttg.async_copy_global_to_local %scale_B, %38 mask %39 : tensor<1x512x!tt.ptr<i8>, #blocked> -> <1x512xi8, #shared1, #ttg.shared_memory, mutable>
      %41 = ttg.async_commit_group %40
      %42 = ttg.async_wait %41 {num = 0 : i32}
      scf.yield %25, %19, %29, %33, %34, %42, %38, %42, %acc_res : i32, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.async.token, tensor<128x128xf32, #blocked1>
    }
    %13 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %1 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<2x128x128xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %scale_A_SMEM : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable>
    ttg.local_dealloc %scale_B_SMEM : !ttg.memdesc<2x1x512xi8, #shared1, #ttg.shared_memory, mutable>
    tt.return %12#8 : tensor<128x128xf32, #blocked1>
  }
}
