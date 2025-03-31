// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load
    tt.func @buffer_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[c_mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[c_mask]]
        // CHECK: %[[aux:.*]] = llvm.mlir.constant(3 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]], {{.*}}, %[[aux]]
        %ret = amdgpu.buffer_load %arg0[%offset] cacheModifier = cs : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask
    tt.func @buffer_load_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        %ret = amdgpu.buffer_load %arg0[%offset], %7 stride = %c256_i32 : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask_other
    tt.func @buffer_load_mask_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        %other = arith.constant dense<0.00e+00> : tensor<128xf32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        // CHECK: llvm.select
        %ret = amdgpu.buffer_load %arg0[%offset], %7, %other stride = %c256_i32: tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store
    tt.func @buffer_store(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: %[[aux:.*]] = llvm.mlir.constant(3 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]], {{.*}}, %[[aux]]
        %c256_i32 = arith.constant 256 : i32
        amdgpu.buffer_store %value, %arg0[%offset] cacheModifier = cs stride = %c256_i32 : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store_mask
    tt.func @buffer_store_mask(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask0:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[mask1:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[mask2:.*]] = llvm.and %[[mask1]], %[[mask0]]
        // CHECK: %[[offset:.*]] = llvm.select %[[mask2]]
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]]
        amdgpu.buffer_store %value, %arg0[%offset], %7 stride = %N : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec4
    tt.func @buffer_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %9 = amdgpu.buffer_load %arg0[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %10 = amdgpu.buffer_load %arg1[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two vectorized store instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xf32>
        amdgpu.buffer_store %11, %arg2[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_load_8xf16
  tt.func public @buffer_load_8xf16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.splat %arg2 : i32 -> tensor<256x64xi32, #blocked>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %4 = arith.addi %3, %1 : tensor<256x64xi32, #blocked>
    // Load 16 f16 elements check for correct vector size of instruction (4xi32 = 8xf16)
    // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xi32>
    %5 = amdgpu.buffer_load %arg0[%4] : tensor<256x64xf16, #blocked>
    // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xi32>
    amdgpu.buffer_store %5, %arg0[%4] : tensor<256x64xf16, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec1
    tt.func @buffer_load_store_vec1(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 elements from A with eight scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %9 = amdgpu.buffer_load %arg0[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %10 = amdgpu.buffer_load %arg1[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two scalar store instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.store {{.*}} : f32
        amdgpu.buffer_store %11, %arg2[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_store_vec2
    tt.func @buffer_load_store_vec2(%arg0: !tt.ptr<f16> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f16>{tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f16>{tt.divisibility = 4: i32}, %arg3: i32{tt.divisibility = 4: i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 fp16 elements from A with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %9 = amdgpu.buffer_load %arg0[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        // Load 8 fp16 elements from B with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %10 = amdgpu.buffer_load %arg1[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf16, #blocked0>
        // Store 8 fp16 elements into C with four i32 scalar store instructionss
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}} : i32
        amdgpu.buffer_store %11, %arg2[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_atomic
    tt.func @buffer_atomic_rmw_fadd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}, %N: i32, %values : tensor<128xf32, #blocked0>, %stride: i32 {tt.divisibility=16:i32}) {
        %c128_i32 = arith.constant 128 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c128_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %mask = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask0:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // There should be a single release fence before any atomics
        // CHECK: llvm.fence syncscope("agent") release
        // CHECK: %[[mask1:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[mask2:.*]] = llvm.and %[[mask1]], %[[mask0]]
        // CHECK: %[[offset:.*]] = llvm.select %[[mask2]]

        // We will have 4 calls to fadd, since the sizePerThread is 4. We should have a vmcnt between each call.
        %ret = amdgpu.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %arg0[%offset], %mask stride = %stride : tensor<128xf32, #blocked0>

        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32

        // There should be a single acquire fence after all of the atomics
        // CHECK: llvm.fence syncscope("agent") acquire
        tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
    // CHECK-LABEL: buffer_load_layout_vectorization
    tt.func public @buffer_load_layout_vectorization(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
        %c1_i32 = arith.constant 1 : i32
        %21 = tt.splat %c1_i32 : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %22 = tt.expand_dims %21 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
        %23 = tt.broadcast %22 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
        // Each thread has to load 8xi16
        // We expect vector size == 1 (i16) for the generated loads as sizePerThread = [1, 1]
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, {{.*}} : i16
        // CHECK-NOT: rocdl.raw.ptr.buffer.load
        %24 = amdgpu.buffer_load %arg0[%23] : tensor<8x16xf16, #blocked>
        tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: strided_buffer_load_and_store
  tt.func public @strided_buffer_load_and_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1024xi32, #blocked>
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %1 = arith.muli %0, %cst : tensor<1024xi32, #blocked>
    // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, {{.*}} : f32
    // CHECK-NOT: rocdl.raw.ptr.buffer.load
    %2 = amdgpu.buffer_load %arg0[%1] : tensor<1024xf32, #blocked>
    // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} : f32
    // CHECK-NOT: rocdl.raw.ptr.buffer.store
    amdgpu.buffer_store %2, %arg1[%1] : tensor<1024xf32, #blocked>
    tt.return
  }
}
