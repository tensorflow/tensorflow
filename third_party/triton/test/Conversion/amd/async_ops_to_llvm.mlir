// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefix=GFX950
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --verify-diagnostics | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy
  tt.func public @async_copy(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per global.load.lds
    // CHECK-COUNT-8: rocdl.global.load.lds
    // CHECK-NOT: rocdl.global.load.lds
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_vectorized_2xf16
  tt.func public @async_copy_vectorized_2xf16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 2 (sizePerThread) per global.load.lds
    // CHECK-COUNT-4: rocdl.global.load.lds
    // CHECK-NOT: rocdl.global.load.lds
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: async_copy_vectorized_8xf16
  tt.func public @async_copy_vectorized_8xf16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per global.load.lds
    // GFX950: rocdl.global.load.lds
    // GFX950-next: llvm.return

    // GFX942 does not support vectorization > 4bytes
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_wait
  tt.func public @async_wait(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                             %arg1: i32 {tt.divisibility = 16 : i32},
                             %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // The waitcnt stores all counters in one i32 bits 15:14 and 3:0 store the vmcnt we have to wait on
    // CHECK: rocdl.s.waitcnt -49168
    // CHECK: rocdl.barrier
    ttg.async_wait {num = 0 : i32}
    // CHECK: rocdl.s.waitcnt -49167
    // CHECK: rocdl.barrier
    ttg.async_wait {num = 1 : i32}
    // CHECK: rocdl.s.waitcnt -2
    // CHECK: rocdl.barrier
    ttg.async_wait {num = 62 : i32}
    // CHECK: rocdl.s.waitcnt -1
    // CHECK: rocdl.barrier
    ttg.async_wait {num = 63 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_commit_group
  tt.func public @async_commit_group(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                     %arg1: i32 {tt.divisibility = 16 : i32},
                                     %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.return
    ttg.async_commit_group
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_mask_other
  tt.func public @async_copy_mask_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
                                %arg3: i32 {tt.divisibility = 16 : i32}) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %29 = arith.addi %arg3, %c31_i32 : i32
    %30 = arith.divsi %29, %c32_i32 : i32
    %31 = arith.cmpi sgt, %30, %c0_i32 : i32

    %51 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %65 = tt.splat %arg3 : i32 -> tensor<32x1xi32, #blocked>
    %66 = arith.cmpi slt, %52, %65 : tensor<32x1xi32, #blocked>
    %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>

    %70 = tt.splat %31 : i1 -> tensor<32x32xi1, #blocked>
    %71 = arith.andi %70, %67 : tensor<32x32xi1, #blocked>

    // Each thread needs to load 4 elements and we load 1 (sizePerThread) per global.load.lds
    // Note that mask/other alignment is 1 so we need 4 conditionals

    // CHECK: llvm.cond_br
    // CHECK: rocdl.global.load.lds
    // CHECK-NEXT: llvm.br
    // CHECK: _predicated_store

    // CHECK: llvm.cond_br
    // CHECK: rocdl.global.load.lds
    // CHECK-NEXT: llvm.br
    // CHECK: _predicated_store

    // CHECK: llvm.cond_br
    // CHECK: rocdl.global.load.lds
    // CHECK-NEXT: llvm.br
    // CHECK: _predicated_store

    // CHECK: llvm.cond_br
    // CHECK: rocdl.global.load.lds
    // CHECK-NEXT: llvm.br
    // CHECK: _predicated_store

    %2 = ttg.async_copy_global_to_local %1, %arg2 mask %67 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [16, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_cache_mods
  tt.func public @async_copy_cache_mods(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    // Each thread needs to load 1 element and we load 1 (sizePerThread) per global.load.lds

    // CHECK: llvm.getelementptr
    // CHECK: %[[aux_ca:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: rocdl.global.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_ca]]
    %2 = ttg.async_copy_global_to_local %1, %arg2 cacheModifier = ca: tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    // CHECK: llvm.getelementptr
    // CHECK: %[[aux_cg:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: rocdl.global.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cg]]
    %3 = ttg.async_copy_global_to_local %1, %arg2 cacheModifier = cg: tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    // CHECK: llvm.getelementptr
    // CHECK: %[[aux_cv:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK: rocdl.global.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cv]]
    %4 = ttg.async_copy_global_to_local %1, %arg2 cacheModifier = cv: tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}
