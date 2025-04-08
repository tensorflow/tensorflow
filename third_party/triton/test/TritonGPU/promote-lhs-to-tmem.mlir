// RUN: env ENABLE_LHS_TO_TMEM=1 triton-opt %s -tritongpu-promote-lhs-to-tmem | FileCheck --dump-input-context=50 %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_lhs
  // CHECK: scf.for
  // CHECK: %[[A:.+]] = tt.load
  // CHECK: %[[A_TMEM:.+]] = ttng.tmem_alloc %[[A]]
  // CHECK: ttng.tc_gen5_mma %[[A_TMEM]]
  tt.func public @promote_lhs(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %acc_res : tensor<128x128xf32, #blocked1>
    }
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %res_f16 : tensor<128x128xf16, #blocked1>
  }

  // CHECK-LABEL: @promote_lhs_mxfp
  // CHECK: scf.for
  // CHECK: %[[A:.+]] = tt.load
  // CHECK: %[[A_TMEM:.+]] = ttng.tmem_alloc %[[A]]
  // CHECK: ttng.tc_gen5_mma_scaled %[[A_TMEM]]
  tt.func public @promote_lhs_mxfp(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32, %a_scale: tensor<128x1xi8, #blocked2>, %b_scale: tensor<64x1xi8, #blocked2>) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %a_scale_tm = ttng.tmem_alloc %a_scale : (tensor<128x1xi8, #blocked2>) -> !ttg.memdesc<128x1xi8, #tmem_scales, #ttng.tensor_memory>
      %b_scale_tm = ttng.tmem_alloc %b_scale : (tensor<64x1xi8, #blocked2>) -> !ttg.memdesc<64x1xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %a_scale_tm, %b_scale_tm, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x1xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<64x1xi8, #tmem_scales, #ttng.tensor_memory>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %acc_res : tensor<128x128xf32, #blocked1>
    }
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %res_f16 : tensor<128x128xf16, #blocked1>
  }

  // CHECK-LABEL: @dont_promote_rhs
  // CHECK: scf.for
  // CHECK: %[[B:.+]] = tt.load
  // CHECK: %[[B_TMEM:.+]] = ttg.local_alloc %[[B]]
  // CHECK: ttng.tc_gen5_mma %{{.+}}, %[[B_TMEM]], %{{.+}}, {{.+}}
  tt.func public @dont_promote_rhs(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %A_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %B = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %A_sh = ttg.memdesc_subview %A_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %acc_res : tensor<128x128xf32, #blocked1>
    }
    ttg.local_dealloc %A_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %res_f16 : tensor<128x128xf16, #blocked1>
  }

  // CHECK-LABEL: @dont_promote_long_lr
  // CHECK: %[[A:.+]] = tt.load
  // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc %[[A]]
  // CHECK: scf.for
  // CHECK: ttng.tc_gen5_mma %[[A_SMEM]]
  tt.func public @dont_promote_long_lr(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %acc_res : tensor<128x128xf32, #blocked1>
    }
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %res_f16 : tensor<128x128xf16, #blocked1>
  }

  // CHECK-LABEL: @dont_convert_layout
  // CHECK: scf.for
  // CHECK: %[[A:.+]] = tt.load
  // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc %[[A]]
  // CHECK: ttng.tc_gen5_mma %[[A_SMEM]]
  tt.func public @dont_convert_layout(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %B_multibuf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B_sh = ttg.memdesc_subview %B_multibuf[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %acc_res : tensor<128x128xf32, #blocked1>
    }
    ttg.local_dealloc %B_multibuf : !ttg.memdesc<1x128x128xf16, #shared, #ttg.shared_memory, mutable>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %res_f16 : tensor<128x128xf16, #blocked1>
  }
}
