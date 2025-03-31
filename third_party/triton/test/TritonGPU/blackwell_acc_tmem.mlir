// RUN: triton-opt %s -split-input-file --tritongpu-keep-acc-in-tmem | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @trivial
  // CHECK: %[[TMEM_BASE:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc %[[TMEM_BASE]]
  // CHECK-NEXT: scf.for
  // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM]]
  // CHECK: scf.yield
  // CHECK: %[[ACC_RES:.*]] = ttng.tmem_load %[[TMEM]]
  // CHECK: %[[RES:.*]] = arith.truncf %[[ACC_RES]]
  // CHECK: tt.return
  tt.func public @trivial(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @use_after_dot
  // CHECK: %[[TMEM_BASE:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc %[[TMEM_BASE]]
  // CHECK-NEXT: scf.for
  // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM]]
  // CHECK: %[[ACC_RES_INS:.*]] = ttng.tmem_load %[[TMEM]]
  // CHECK: arith.addf %[[ACC_RES_INS]]
  // CHECK: scf.yield
  // CHECK: %[[ACC_RES:.*]] = ttng.tmem_load %[[TMEM]]
  // CHECK: %[[RES:.*]] = arith.truncf %[[ACC_RES]]
  // CHECK: tt.return
  tt.func public @use_after_dot(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %add: tensor<128x128xf32, #blocked>, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %add_ = %cst) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %add_res = arith.addf %acc_res, %add_ : tensor<128x128xf32, #blocked>
      scf.yield %acc_res, %add_res : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res#0 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @use_before_dot
  // CHECK: %[[TMEM_BASE:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc %[[TMEM_BASE]]
  // CHECK-NEXT: scf.for
  // CHECK: %[[ACC_RES_INS:.*]] = ttng.tmem_load %[[TMEM]]
  // CHECK: arith.addf %[[ACC_RES_INS]]
  // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM]]
  // CHECK: scf.yield
  // CHECK: %[[ACC_RES:.*]] = ttng.tmem_load %[[TMEM]]
  // CHECK: %[[RES:.*]] = arith.truncf %[[ACC_RES]]
  // CHECK: tt.return
  tt.func public @use_before_dot(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %add: tensor<128x128xf32, #blocked>, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res:2 = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst, %add_ = %cst) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>)  : i32 {
      %add_res = arith.addf %acc, %add_ : tensor<128x128xf32, #blocked>
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_tm = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res, %add_res : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res#0 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_constant_inputs
  tt.func public @hoist_constant_inputs(%arg0: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem>, %arg2: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, %arg3: i32, %arg4: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: arith.trunci
    // CHECK: tt.splat
    // CHECK: ttng.tmem_alloc
    // CHECK: scf.for
    // CHECK:  ttng.tc_gen5_mma_scaled
    scf.for %arg5 = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %0 = arith.trunci %arg3 : i32 to i8
      %1 = tt.splat %0 : i8 -> tensor<128x4xi8, #blocked1>
      %2 = ttng.tmem_alloc %1 : (tensor<128x4xi8, #blocked1>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %arg0, %arg1, %arg4, %arg2, %2, %true, %true lhs = e5m2 rhs = e2m1 : (!ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<64x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, i1, i1) -> ()
    }
    tt.return
  }
}
