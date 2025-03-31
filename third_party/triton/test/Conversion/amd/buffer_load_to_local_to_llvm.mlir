// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefixes=COMMON,GFX950
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --verify-diagnostics | FileCheck %s --check-prefixes=COMMON,GFX942

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_simple
  tt.func public @buffer_load_to_local_simple(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f16>,
                                %arg2: tensor<32x64xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per buffer load instruction
    // COMMON: rocdl.make.buffer.rsrc
    // COMMON-NOT: rocdl.make.buffer.rsrc
    // COMMON-COUNT-8: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    %65 = amdgpu.buffer_load_to_local %arg1[%arg2] into %arg3 {OpIdx = #amdgpu.OpIdx<1>} : <f16>[tensor<32x64xi32, #blocked>] -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 32], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_vectorized_2xf16
  tt.func public @buffer_load_to_local_vectorized_2xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %3, %6 : tensor<64x64xi32, #blocked>

    // Each thread needs to load 2 elements and we load 2 (sizePerThread) per buffer load instruction
    // COMMON: rocdl.make.buffer.rsrc
    // COMMON-NOT: rocdl.make.buffer.rsrc
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    %8 = amdgpu.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<64x64xi32, #blocked>]  -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 32], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_vectorized_8xf16
  tt.func public @buffer_load_to_local_vectorized_8xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %3, %6 : tensor<64x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per buffer load instruction
    // GFX950: rocdl.make.buffer.rsrc
    // GFX950-NOT: rocdl.make.buffer.rsrc
    // GFX950: rocdl.raw.ptr.buffer.load.lds
    // GFX950-NOT: rocdl.raw.ptr.buffer.load.lds

    // GFX942 does not support vectorization > 4bytes so we cannot lower it
    // GFX942-NOT: rocdl.raw.ptr.buffer.load.lds
    // GFX942: amdgpu.buffer_load_to_local
    %8 = amdgpu.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<64x64xi32, #blocked>]  -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_mask_other
  tt.func public @buffer_load_to_local_mask_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f16>,
                                %arg2: tensor<32x32xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
                                %arg4: i32) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %29 = arith.addi %arg4, %c31_i32 : i32
    %30 = arith.divsi %29, %c32_i32 : i32
    %31 = arith.cmpi sgt, %30, %c0_i32 : i32

    %51 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %65 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %66 = arith.cmpi slt, %52, %65 : tensor<32x1xi32, #blocked>
    %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>

    %70 = tt.splat %31 : i1 -> tensor<32x32xi1, #blocked>
    %71 = arith.andi %70, %67 : tensor<32x32xi1, #blocked>

    // Each thread needs to load 4 elements and we load 1 (sizePerThread) per buffer load instruction
    // Note that mask/other alignment is 1 so we need 4 conditionals

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: _predicated_store

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: _predicated_store

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: _predicated_store

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: _predicated_store

    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: _predicated_store

    amdgpu.buffer_load_to_local %arg1[%arg2] mask=%67 other=%cst_0 into %arg3 {OpIdx = #amdgpu.OpIdx<1>} : <f16>[tensor<32x32xi32, #blocked>] tensor<32x32xf16, #blocked>  -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_cache_mods
  tt.func public @buffer_load_to_local_cache_mods(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    // The first constant 0 skips the LDS offset which is also 0
    // COMMON: llvm.getelementptr
    // COMMON: llvm.mlir.constant(0 : i32) : i32
    // COMMON: %[[aux_ca:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_ca]]
    %1 = amdgpu.buffer_load_to_local %arg0[%0] cacheModifier = ca into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    // COMMON: llvm.getelementptr
    // COMMON: %[[aux_cg:.*]] = llvm.mlir.constant(3 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cg]]
    %2 = amdgpu.buffer_load_to_local %arg0[%0] cacheModifier = cg into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    // COMMON: llvm.getelementptr
    // COMMON: %[[aux_cv:.*]] = llvm.mlir.constant(17 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cv]]
    %3 = amdgpu.buffer_load_to_local %arg0[%0] cacheModifier = cv into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>

    tt.return
  }
}
