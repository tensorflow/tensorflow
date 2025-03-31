// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.store
    // CHECK: llvm.br
    // CHECK: rocdl.barrier
    // CHECK: llvm.load
    // CHECK: llvm.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.store
    // CHECK: llvm.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// Smoke test to check that mfma 32 and dot operand layouts can work with small tensors, for example with shape 16x16
#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
#dotop1 = #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth=4}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: small_mfma_tensor_conversions
  tt.func public @small_mfma_tensor_conversions(%arg0: tensor<16x16xf16, #mfma>, %arg1: tensor<16x16x!tt.ptr<f32>, #mfma>) {
    // CHECK-NOT: ttg.convert_layout
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #mfma>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    // CHECK-4: store {{.*}} vector<4xf16>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #dotop0>
    // CHECK-2: load {{.*}} vector<4xf16>
    %2 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #dotop1>
    // CHECK-8: load {{.*}} vector<1xf16>
    %3 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #mfma>
    // CHECK-4: load {{.*}} vector<4xf16>
    %4 = tt.fp_to_fp %3 : tensor<16x16xf16, #mfma> -> tensor<16x16xf32, #mfma>

    %5 = tt.dot %1, %2, %4 : tensor<16x16xf16, #dotop0> * tensor<16x16xf16, #dotop1> -> tensor<16x16xf32, #mfma>
    // Store result to prevent DCE from removing all conversion related code
    %6 = ttg.local_alloc %5 : (tensor<16x16xf32, #mfma>) -> !ttg.memdesc<16x16xf32, #shared, #smem>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f16x2
  tt.func @atomic_add_f16x2(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked1>, %arg2 : tensor<256xf16, #blocked1>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %base_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked1>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xi32, #blocked1>
    // CHECK: llvm.cond_br
    // CHECK-NOT: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xf16>
    // CHECK-NOT: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xf16, #blocked1>, tensor<256xi1, #blocked1>) -> tensor<256xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_bf16x2
  tt.func @atomic_add_bf16x2(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked2>, %arg2 : tensor<256xbf16, #blocked2>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %base_ptr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked2>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xi32, #blocked2>
    // CHECK: llvm.cond_br
    // CHECK-NOT: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xbf16>
    // CHECK-NOT: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xbf16, #blocked2>, tensor<256xi1, #blocked2>) -> tensor<256xbf16, #blocked2>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f16_dpp
  tt.func @atomic_add_f16_dpp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked1>, %arg2 : tensor<256xf16, #blocked1>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %base_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked1>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xi32, #blocked1>
    // CHECK: llvm.cond_br
    // CHECK: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xf16>
    // CHECK: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xf16, #blocked1>, tensor<256xi1, #blocked1>) -> tensor<256xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_bf16_dpp
  tt.func @atomic_add_bf16_dpp(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked2>, %arg2 : tensor<256xbf16, #blocked2>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %base_ptr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked2>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xi32, #blocked2>
    // CHECK: llvm.cond_br
    // CHECK: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xbf16>
    // CHECK: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xbf16, #blocked2>, tensor<256xi1, #blocked2>) -> tensor<256xbf16, #blocked2>
    tt.return
  }
}

// -----

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<64xf32, #blocked3>) {
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 322, 10, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 323, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK: llvm.amdgcn.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64xf32, #blocked3>) -> f32
    tt.return
  }
}

// -----

#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_xor_max
  tt.func @reduce_xor_max(%arg0: tensor<32xf32, #blocked4>) {
    // CHECK: rocdl.ds_swizzle
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 12, false : i32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 264, 15, 3, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 10, false : i32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 260, 15, 5, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 78, 15, 15, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 177, 15, 15, false : i32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked4>) -> f32
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomicrmw_scope_memsemantics
  tt.func @atomicrmw_scope_memsemantics(%arg0 : tensor<128x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<128xi1, #blocked0>, %arg2 : tensor<128xf32, #blocked0>) {
    // relaxed
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} monotonic
    %0 = tt.atomic_rmw fadd, relaxed, sys, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"agent"}}) monotonic
    %1 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"workgroup"}}) monotonic
    %2 = tt.atomic_rmw fadd, relaxed, cta, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>

    // acquire
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} acquire
    %3 = tt.atomic_rmw fadd, acquire, sys, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"agent"}}) acquire
    %4 = tt.atomic_rmw fadd, acquire, gpu, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"workgroup"}}) acquire
    %5 = tt.atomic_rmw fadd, acquire, cta, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>

    // release
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} release
    %6 = tt.atomic_rmw fadd, release, sys, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"agent"}}) release
    %7 = tt.atomic_rmw fadd, release, gpu, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"workgroup"}}) release
    %8 = tt.atomic_rmw fadd, release, cta, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>

    // acq_rel
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} acq_rel
    %9 = tt.atomic_rmw fadd, acq_rel, sys, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"agent"}}) acq_rel
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>
    // CHECK: llvm.atomicrmw {{.*}}, {{.*}} syncscope({{"workgroup"}}) acq_rel
    %11 = tt.atomic_rmw fadd, acq_rel, cta, %arg0, %arg2, %arg1 : (tensor<128x!tt.ptr<f32>, #blocked0>, tensor<128xf32, #blocked0>, tensor<128xi1, #blocked0>) -> tensor<128xf32, #blocked0>

    tt.return
  }
}

// -----

#blocked5 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: atomic_runtime_lds_reduction
  tt.func @atomic_runtime_lds_reduction(%arg0 : tensor<64x!tt.ptr<f32>, #blocked5>, %arg2 : tensor<64xf32, #blocked5>) {

    // CHECK: llvm.zext
    // CHECK-COUNT-7: rocdl.update.dpp
    // CHECK: llvm.bitcast
    // CHECK-COUNT: llvm.amdgcqn.ds.permute
    // CHECK: llvm.bitcast
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.bitcast
    // CHECK-COUNT-2: llvm.amdgcn.ds.permute
    // CHECK: llvm.bitcast
    // CHECK: llvm.inttoptr
    // CHECK: llvm.amdgcn.ballot
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.amdgcn.ballot

    // loop body:
    // CHECK: llvm.bitcast
    // CHECK-COUNT-2: llvm.amdgcn.readfirstlane
    // CHECK: llvm.bitcast
    // CHECK: llvm.amdgcn.ballot
    // CHECK: rocdl.mbcnt.lo
    // CHECK: rocdl.mbcnt.hi

    // share info:
    // 1. address
    // CHECK: llvm.bitcast
    // CHECK-COUNT-2: llvm.amdgcn.ds.permute
    // CHECK: llvm.bitcast
    // 2. value
    // CHECK: llvm.amdgcn.ds.permute
    // CHECK: llvm.bitcast
    // 3. packed methadata
    // CHECK: llvm.bitcast
    // CHECK: llvm.amdgcn.ds.permute
    // CHECK: llvm.bitcast

    // CHECK: llvm.amdgcn.ballot

    // reduction:
    // CHECK-COUNT-6: llvm.amdgcn.ds.bpermute

    // CHECK: inttoptr
    // CHECK: llvm.atomicrmw
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2 {allocation.offset = 0 : i32} : (tensor<64x!tt.ptr<f32>, #blocked5>, tensor<64xf32, #blocked5>) -> tensor<64xf32, #blocked5>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_i8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_i8(%arg0: tensor<16x16xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xi32, #blocked>) {
    // CHECK-4: llvm.call_intrinsic "llvm.amdgcn.sdot4"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xi32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_fp16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_fp16(%arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK-8: llvm.call_intrinsic "llvm.amdgcn.fdot2"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_fp16_fp16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_fp16_fp16(%arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf16, #blocked>) {
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.fmuladd.f16"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: amd_rotating_shared_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.amd_rotating_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_rotating_shared_layout(%arg0: tensor<64x64xf16, #blocked>) {
    // CHECK-COUNT-16: llvm.store {{.*}} : vector<1xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-16: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = ttg.local_load %0 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    // CHECK-COUNT-16: llvm.store {{.*}} : vector<1xf16>, !llvm.ptr<3>
    ttg.local_store %1, %0 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: amd_rotating_subview_shared_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.amd_rotating_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_rotating_subview_shared_layout(%arg0: tensor<64x64xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    // CHECK-COUNT-16: llvm.store {{.*}} : vector<1xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c16_i32, %c0_i32] : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 64x64>
    // CHECK-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %2 = ttg.local_load %1 : !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 64x64> -> tensor<16x64xf16, #blocked>
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<1xf16>, !llvm.ptr<3>
    ttg.local_store %2, %1 : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 64x64>
    tt.return
  }
}
