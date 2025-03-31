// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i32, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK: nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK:  llvm.return
    tt.return
  }
} // end module

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  tt.func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mov.u32 $0, $1;
    // CHECK-SAME: @$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"
    // CHECK: llvm.inline_asm
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load
  tt.func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: vectorized_load_f16
  tt.func @vectorized_load_f16(%a_ptr_init: tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other
  tt.func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other_vec
  tt.func @masked_load_const_other_vec(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_with_cache_attr
  tt.func @store_with_cache_attr(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    //      CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.L1::evict_last.b32
    //      CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.L1::evict_last.b32
    tt.store %a_ptr_init, %cst_0, %cst evictionPolicy = evict_last cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_no_vec
  tt.func @global_load_store_no_vec(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from vector0
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from vector1
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global
    // CHECK: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_vec4
  tt.func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// This test verifies the vectorization of Load and Store Ops.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// Note, the %n_elements doesn't have a "tt.divisibility" hint, so Triton assumes it's divisibility is 1, this should effect the mask's alignment and further restrict the load/store ops' vector width to be 1.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  tt.func @vecadd_masked_vec1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %9 = tt.splat %n_elements : i32 -> tensor<64xi32, #blocked>
    %10 = arith.cmpi "slt", %4, %9 : tensor<64xi32, #blocked>
    // load op has a vector width = 1 due to the %mask's alignment
    // CHECK: ld.global.b32
    %11 = tt.load %6, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %12 = tt.load %8, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %11, %12 : tensor<64xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %15, %13, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with two vectorized load instruction
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with two vectorized load instruction
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with two vectorized store instruction
    // CHECK: st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    // CHECK: st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// Slice layout with 2 unique elements, but 8 total elements per thread
#blocked2d = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked2d}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_slice
  tt.func @global_load_store_slice(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #slice>
    %4 = arith.addi %3, %2 : tensor<128xi32, #slice>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #slice>
    %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>, #slice>, tensor<128xi32, #slice>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #slice>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #slice>, tensor<128xi32, #slice>

    // Load 2 element from vector0 without predicate
    // CHECK: mov.u32 $0, 0x0
    // CHECK-NOT: @{{.*}} ld.global
    // CHECK-COUNT-2: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 2 elements from vector1 without predicate
    // CHECK: mov.u32 $0, 0x0
    // CHECK-NOT: @{{.*}} ld.global
    // CHECK-COUNT-2: ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    %9 = tt.load %6 : tensor<128x!tt.ptr<f32>, #slice>
    %10 = tt.load %8 : tensor<128x!tt.ptr<f32>, #slice>
    %11 = arith.addf %9, %10 : tensor<128xf32, #slice>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #slice>
    %13 = tt.addptr %12, %4 : tensor<128x!tt.ptr<f32>, #slice>, tensor<128xi32, #slice>

    // Store 2 element to global without predicate
    // CHECK-NOT: @{{.*}} st.global
    // CHECK-COUNT-2: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %13, %11 : tensor<128x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// TODO: Add a testcase to verify the optimization when ptr of the LoadOp
//       is from an addptr with const idx

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_view_broadcast
  tt.func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.mlir.undef
    // CHECK: %[[T0:.*]] = llvm.extractvalue
    // CHECK: %[[T1:.*]] = llvm.extractvalue
    %0 = tt.reshape %arg allow_reorder : tensor<256xf32, #blocked0> -> tensor<256x1xf32,#blocked2>
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    %1 = tt.broadcast %0 : tensor<256x1xf32,#blocked2> -> tensor<256x4xf32, #blocked2>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: basic_make_range
  tt.func @basic_make_range() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}


// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: sliced_layout_make_range
  tt.func @sliced_layout_make_range() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked0}>>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addf
  tt.func @basic_addf(%arg0 : tensor<256xf32,#blocked0>, %arg1 : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.fadd
    // CHECK: llvm.fadd
    %1 = arith.addf %arg0, %arg1 : tensor<256xf32,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addi
  tt.func @basic_addi(%arg0 : tensor<256xi32,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.add
    // CHECK: llvm.add
    %1 = arith.addi %arg0, %arg1 : tensor<256xi32,#blocked0>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  tt.func @basic_program_id() {
    // CHECK: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.ctaid.x"() : () -> i32
    %0 = tt.get_program_id x : i32
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addptr
  tt.func @basic_addptr(%arg0 : tensor<256x!tt.ptr<f32>,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.getelementptr
    %0 = tt.addptr %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_alloc_tensor
  tt.func @basic_alloc_tensor() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NEXT: llvm.getelementptr
    // CHECK-NEXT: llvm.mlir.constant
    %0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_subview
  tt.func @basic_subview() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: llvm.mlir.constant(32 : i32) : i32
    // CHECK-NEXT: llvm.mlir.constant(512 : i32) : i32
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.undef
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x16x32xf32, #shared0, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%index, %zero, %zero] : !ttg.memdesc<128x16x32xf32, #shared0, #smem, mutable> -> !ttg.memdesc<16x32xf32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: nvmma_subview
  tt.func @nvmma_subview() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.mlir.undef : i32
    // CHECK-NEXT: llvm.mlir.constant(32 : i32) : i32
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.sub
    // CHECK-NEXT: llvm.udiv
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.sub
    // CHECK-NEXT: llvm.urem
    // CHECK-NEXT: llvm.sub
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.udiv
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.urem
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<16x128xf32, #shared0, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%zero, %zero] : !ttg.memdesc<16x128xf32, #shared0, #smem, mutable> -> !ttg.memdesc<16x32xf32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_async_wait
  tt.func @basic_async_wait() {
    // CHECK: nvvm.cp.async.wait.group 4
    ttg.async_wait {num = 4: i32}
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#slice1d0 = #ttg.slice<{dim = 0, parent = #blocked1}>
#shared1D = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 8, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared2D = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_1d
  tt.func @basic_insert_slice_async_1d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<64> : tensor<64xi32, #slice1d0>
    %58 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>, #slice1d0>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1d0>
    %59 = tt.addptr %58, %24 : tensor<64x!tt.ptr<i64>, #slice1d0>, tensor<64xi32, #slice1d0>
    %66 = tt.addptr %59, %cst_2 : tensor<64x!tt.ptr<i64>, #slice1d0>, tensor<64xi32, #slice1d0>
    %71 = ttg.local_alloc : () -> !ttg.memdesc<2x64xi64, #shared2D, #smem, mutable>
    %subview = ttg.memdesc_subview %71[%c0_i32, %c0_i32] :
      !ttg.memdesc<2x64xi64, #shared2D, #smem, mutable> ->
      !ttg.memdesc<64xi64, #shared1D, #smem, mutable>
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: nvvm.cp.async.commit.group
    %73 = ttg.async_copy_global_to_local %66, %subview : tensor<64x!tt.ptr<i64>, #slice1d0> -> !ttg.memdesc<64xi64, #shared1D, #smem, mutable>
    ttg.async_commit_group %73
    tt.return
  }
}

// -----

#block0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #ttg.slice<{dim = 1, parent=#block2}>
#slice3d0 = #ttg.slice<{dim = 0, parent=#block3}>
#AL = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v4
  tt.func @basic_insert_slice_async_v4(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<16xi32, #slice2d1> -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<64xi32, #slice3d0> -> tensor<1x64xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<16x1xi32, #block2> -> tensor<16x64xi32, #block2>
    %cst_scalar = arith.constant 64 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<16x64xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x64xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x64xi32, #block3> -> tensor<16x64xi32, #block3>
    %broadcast_off0 = ttg.convert_layout %broadcast_off0_ : tensor<16x64xi32, #block2> -> tensor<16x64xi32, #AL>
    %broadcast_off1 = ttg.convert_layout %broadcast_off1_ : tensor<16x64xi32, #block3> -> tensor<16x64xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x64xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x64x!tt.ptr<f32>, #AL>, tensor<16x64xi32, #AL>
    %tensor = ttg.local_alloc : () -> !ttg.memdesc<16x64xf32, #A, #smem, mutable>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10;"
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "cp.async.cg.shared.global [ ${{.*}} + 16 ], [ ${{.*}} + 0 ], 0x10, 0x10;"
    // CHECK: nvvm.cp.async.commit.group
    %a = ttg.async_copy_global_to_local %a_ptr, %tensor : tensor<16x64x!tt.ptr<f32>, #AL> -> !ttg.memdesc<16x64xf32, #A, #smem, mutable>
    ttg.async_commit_group
    tt.return
  }
}

// -----

#block0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #ttg.slice<{dim = 1, parent=#block2}>
#slice3d0 = #ttg.slice<{dim = 0, parent=#block3}>
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1
  tt.func @basic_insert_slice_async_v1(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<16xi32, #slice2d1> -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<32xi32, #slice3d0> -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<16x1xi32, #block2> -> tensor<16x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<16x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x32xi32, #block3> -> tensor<16x32xi32, #block3>
    %broadcast_off0 = ttg.convert_layout %broadcast_off0_ : tensor<16x32xi32, #block2> -> tensor<16x32xi32, #AL>
    %broadcast_off1 = ttg.convert_layout %broadcast_off1_ : tensor<16x32xi32, #block3> -> tensor<16x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x32xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x32x!tt.ptr<f32>, #AL>, tensor<16x32xi32, #AL>
    %tensor = ttg.local_alloc : () -> !ttg.memdesc<16x32xf32, #A, #smem, mutable>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: nvvm.cp.async.commit.group
    %a = ttg.async_copy_global_to_local %a_ptr, %tensor : tensor<16x32x!tt.ptr<f32>, #AL> -> !ttg.memdesc<16x32xf32, #A, #smem, mutable>
    ttg.async_commit_group
    tt.return
  }
}

// -----

#block0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #ttg.slice<{dim = 1, parent=#block2}>
#slice3d0 = #ttg.slice<{dim = 0, parent=#block3}>
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1_multictas
  tt.func @basic_insert_slice_async_v1_multictas(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<32xi32, #slice2d1> -> tensor<32x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<32xi32, #slice3d0> -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<32x1xi32, #block2> -> tensor<32x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<32x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<32x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x32xi32, #block3> -> tensor<32x32xi32, #block3>
    %broadcast_off0 = ttg.convert_layout %broadcast_off0_ : tensor<32x32xi32, #block2> -> tensor<32x32xi32, #AL>
    %broadcast_off1 = ttg.convert_layout %broadcast_off1_ : tensor<32x32xi32, #block3> -> tensor<32x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<32x32xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %tensor = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #A, #smem, mutable>
    %index = arith.constant 1 : i32

    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.mul
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4;"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: nvvm.cp.async.commit.group
    %a = ttg.async_copy_global_to_local %a_ptr, %tensor : tensor<32x32x!tt.ptr<f32>, #AL> -> !ttg.memdesc<32x32xf32, #A, #smem, mutable>
    ttg.async_commit_group
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: basic_splat
  tt.func @basic_splat(%ptr: !tt.ptr<f32>) {
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_store
  tt.func @basic_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked0>, %vals: tensor<256xf32, #blocked0>, %mask: tensor<256xi1, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked
  tt.func @convert_layout_blocked_blocked(%arg0: tensor<32x32xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-COUNT-8: llvm.inline_asm {{.*}} st.shared
    // CHECK-: nvvm.barrier0
    // CHECK-COUNT-8: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked0> -> tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked_vec
  tt.func @convert_layout_blocked_blocked_vec(%arg0: tensor<32x32xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.inline_asm
    // CHECK: st.shared
    // CHECK: llvm.inline_asm
    // CHECK: st.shared
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked0> -> tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {

// CHECK-LABEL: convert_layout_ptr_element
tt.func @convert_layout_ptr_element(%arg0: tensor<16x16x!tt.ptr<i32>, #blocked0>) {
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.inttoptr
  %0 = ttg.convert_layout %arg0 : tensor<16x16x!tt.ptr<i32>, #blocked0> -> tensor<16x16x!tt.ptr<i32>, #blocked2>
  tt.return
}

}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep
  tt.func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.inline_asm
    // CHECK: st.shared
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK: llvm.load
    // CHECK: nvvm.barrier0
    // CHECK: llvm.inline_asm
    // CHECK: st.shared
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_ldmatrix
  tt.func @convert_dot_ldmatrix(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    // CHECK: nvgpu.ldmatrix %{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: nvgpu.ldmatrix %{{.*}} {trans} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    // CHECK: nvgpu.ldmatrix %{{.*}} {trans} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    // CHECK-NOT: nvgpu.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 8, perPhase=1, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  tt.func @convert_dot_ldmatrix_swizzle(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    // CHECK: nvgpu.ldmatrix %{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: nvgpu.ldmatrix %{{.*}} {trans} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    // CHECK: nvgpu.ldmatrix %{{.*}} {trans} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    // CHECK-NOT: nvgpu.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase=1, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  tt.func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    // CHECK-NOT: nvgpu.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mmav3_shared
  tt.func @convert_dot_mmav3_shared(%A: tensor<64x64xf16, #blocked0>, %B: tensor<64x64xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<64x64xf16, #blocked0>) -> !ttg.memdesc<64x64xf16, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<64x64xf16, #blocked0>) -> !ttg.memdesc<64x64xf16, #shared0, #smem>
    // CHECK-NOT: nvgpu.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<64x64xf16, #shared0, #smem> -> tensor<64x64xf16, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<64x64xf16, #shared0, #smem> -> tensor<64x64xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma0>

    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<64x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<64x64xf32, #mma0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 16, perPhase=1, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=4}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=4}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_fp8
  tt.func @convert_dot_fp8(%A: tensor<16x16xf8E5M2, #blocked0>, %B: tensor<16x16xf8E5M2, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf8E5M2, #blocked0>) -> !ttg.memdesc<16x16xf8E5M2, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<16x16xf8E5M2, #blocked0>) -> !ttg.memdesc<16x16xf8E5M2, #shared0, #smem>
    // CHECK: nvgpu.ldmatrix %{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    // CHECK-NOT: nvgpu.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf8E5M2, #shared0, #smem> -> tensor<16x16xf8E5M2, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<16x16xf8E5M2, #shared0, #smem> -> tensor<16x16xf8E5M2, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf8E5M2, #dot_operand_a> * tensor<16x16xf8E5M2, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_mmav2_block
  tt.func @convert_layout_mmav2_blocked(%arg0: tensor<32x16xf32, #mma>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.shared
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.shared
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<32x16xf32, #mma> -> tensor<32x16xf32, #blocked0>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot1 = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_layout_mmav2_dot_reg
  tt.func @convert_layout_mmav2_dot_reg(%arg0: tensor<16x16xf16, #mma>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #dot1>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot1 = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_layout_mmav2_dot_reg
  tt.func @convert_layout_mmav2_dot_reg(%arg0: tensor<1x16xf16, #mma>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<1x16xf16, #mma> -> tensor<1x16xf16, #dot1>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#slice = #ttg.slice<{dim = 0, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_slice_mmav2_blocked_reg
  tt.func @convert_layout_slice_mmav2_blocked_reg(%arg0: tensor<1xf16, #slice>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<1xf16, #slice> -> tensor<1xf16, #blocked>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_0
  tt.func @convert_layout_mmav3_mmav3_0(%arg0: tensor<64x64xf16, #mma0>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #mma0> -> tensor<64x64xf16, #mma1>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_1
  tt.func @convert_layout_mmav3_mmav3_1(%arg0: tensor<64x64xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #mma1> -> tensor<64x64xf16, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_2
  tt.func @convert_layout_mmav3_mmav3_2(%arg0: tensor<16x16xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mma1> -> tensor<16x16xf16, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_3
  tt.func @convert_layout_mmav3_mmav3_3(%arg0: tensor<1x64xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<1x64xf16, #mma1> -> tensor<1x64xf16, #mma0>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot1 = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_layout_mmav2_dot_reg
  tt.func @convert_layout_mmav2_dot_reg(%arg0: tensor<16x16xf16, #mma>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #dot1>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_0
  tt.func @convert_layout_mmav3_mmav3_0(%arg0: tensor<64x64xf16, #mma0>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #mma0> -> tensor<64x64xf16, #mma1>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_1
  tt.func @convert_layout_mmav3_mmav3_1(%arg0: tensor<64x64xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #mma1> -> tensor<64x64xf16, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_2
  tt.func @convert_layout_mmav3_mmav3_2(%arg0: tensor<16x16xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mma1> -> tensor<16x16xf16, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_mmav3_mmav3_3
  tt.func @convert_layout_mmav3_mmav3_3(%arg0: tensor<1x64xf16, #mma1>) {
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    %0 = ttg.convert_layout %arg0 : tensor<1x64xf16, #mma1> -> tensor<1x64xf16, #mma0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_mmav3_transpose
  tt.func @convert_layout_mmav3_transpose(%arg0: tensor<128x256xf8E5M2, #mma>) {
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    // CHECK-COUNT-16: st.shared.b8
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    %0 = ttg.convert_layout %arg0 : tensor<128x256xf8E5M2, #mma> -> tensor<128x256xf8E5M2, #blocked>
    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_shared
  tt.func @convert_layout_blocked_shared(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !ttg.memdesc<128x32xf32, #shared0, #smem>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice0
  tt.func @convert_blocked1d_to_slice0(%src:tensor<32xi32, #blocked0>) {
    // CHECK: llvm.load {{.*}} -> vector<4xi32>
    %cvt = ttg.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice1
  tt.func @convert_blocked1d_to_slice1(%src:tensor<32xi32, #blocked0>) {
    // CHECK-COUNT-8: llvm.load {{.*}} -> i32
    %cvt = ttg.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked_to_blocked_ptr
  tt.func @convert_blocked_to_blocked_ptr(%src:tensor<32x!tt.ptr<f32>, #blocked0>) {
    // CHECK: llvm.ptrtoint
    // CHECK: inline_asm{{.*}}st.shared
    // CHECK: nvvm.barrier0
    // CHECK: llvm.inttoptr
    // CHECK-COUNT-4: llvm.insertvalue
    %cvt = ttg.convert_layout %src : tensor<32x!tt.ptr<f32>, #blocked0> -> tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Regression test for https://github.com/triton-lang/triton/issues/5745
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [[1, 0], [2, 0], [4, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 2]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0], [0, 1]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: linear_layout_with_multiple_iterations
  tt.func @linear_layout_with_multiple_iterations(%src: tensor<8x4xbf16, #linear>) {
    %cvt = ttg.convert_layout %src : tensor<8x4xbf16, #linear> -> tensor<8x4xbf16, #linear1>
    // CHECK: inline_asm{{.*}}st.shared.v2
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK: nvvm.barrier0
    // CHECK: inline_asm{{.*}}st.shared.v2
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!ttg.memdesc<128x32xf16, #shared, #smem>, %b:!ttg.memdesc<32x256xf16, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    // CHECK: nvgpu.ldmatrix
    %a_mat = ttg.local_load %a : !ttg.memdesc<128x32xf16, #shared, #smem> -> tensor<128x32xf16, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<32x256xf16, #shared, #smem> -> tensor<32x256xf16, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst : tensor<128x32xf16, #dot_operand_a> * tensor<32x256xf16, #dot_operand_b> -> tensor<128x256xf32, #mma>
    %38 = ttg.convert_layout %28 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_fmadot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!ttg.memdesc<32x16xf32, #shared, #smem>, %b:!ttg.memdesc<16x32xf32, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // CHECK: llvm.intr.fmuladd
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xf32, #shared, #smem> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = ieee : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=1}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!ttg.memdesc<32x16xf32, #shared, #smem>, %b:!ttg.memdesc<16x32xf32, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: nvgpu.ldmatrix
    // CHECK-SAME: (i32, i32, i32, i32)
    // CHECK: nvgpu.ldmatrix
    // CHECK-SAME: (i32, i32, i32, i32)
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xf32, #shared, #smem> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #dot_operand_b>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #mma>
    %38 = ttg.convert_layout %28 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32_sys_scope(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.sys.relaxed.add.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.sys.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, sys, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f16_nomask(%dest_ptrs: tensor<256x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf16, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f16_nomask
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f16_withmask(%dest_ptrs: tensor<256x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf16, #blocked>, %mask: tensor<256xi1, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f16_withmask
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32
  tt.func @store_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32
    tt.store %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32_scalar
  tt.func @store_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : f32) {
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    tt.store %arg0, %arg1 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z: i32
  // CHECK: ctaid.x
  // CHECK: ctaid.y
  // CHECK: ctaid.z
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [4], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z : i32
  // CHECK: clusterid.x
  // CHECK: clusterid.y
  // CHECK: clusterid.z
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_get_num_program
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    %blockdimx = tt.get_num_programs x : i32
    %blockdimy = tt.get_num_programs y : i32
    %blockdimz = tt.get_num_programs z : i32
    // CHECK: nctaid.x
    // CHECK: nctaid.y
    // CHECK: nctaid.z
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [4], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    %blockdimx = tt.get_num_programs x : i32
    %blockdimy = tt.get_num_programs y : i32
    %blockdimz = tt.get_num_programs z : i32
    // CHECK: nclusterid.x
    // CHECK: nclusterid.y
    // CHECK: nclusterid.z
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_index_cache
  tt.func @test_index_cache() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_base_index_cache
  tt.func @test_base_index_cache(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = ttg.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !ttg.memdesc<128x32xf32, #shared0, #smem>
    %1 = ttg.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !ttg.memdesc<128x32xf32, #shared0, #smem>
    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_index_cache_different_block
  tt.func @test_index_cache_different_block(%arg0: tensor<128x32xf32, #blocked0>, %arg1: i1) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = ttg.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !ttg.memdesc<128x32xf32, #shared0, #smem>
    cf.cond_br %arg1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = ttg.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !ttg.memdesc<128x32xf32, #shared0, #smem>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32_cst_b
  tt.func @matmul_tf32_cst_b(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a: tensor<32x16xf32, #dot_operand_a>, %c: tensor<32x32xf32, #mma>) {
  // CHECK: %[[CST:.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[BC:.+]] = llvm.bitcast %[[CST]] : f32 to f32
  // CHECK: %[[SI:.+]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: llvm.insertvalue %[[BC]], %[[SI]][0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    %b_mat = arith.constant dense<1.000000e+00> : tensor<16x32xf32, #dot_operand_b>
    %28 = tt.dot %a, %b_mat, %c, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #mma>
    %38 = ttg.convert_layout %28 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_f16_cst_operands
  tt.func public @matmul_f16_cst_operands(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
  // CHECK: %[[U:.+]] = llvm.mlir.undef : vector<2xf16>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[V0:.+]] = llvm.insertelement %{{.*}}, %[[U]][%[[C0]] : i32] : vector<2xf16>
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[V1:.+]] = llvm.insertelement %{{.*}}, %[[V0]][%[[C1]] : i32] : vector<2xf16>
  // CHECK: %[[BC:.+]] = llvm.bitcast %[[V1]] : vector<2xf16> to i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = arith.muli %3, %cst_2 : tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %9 = tt.broadcast %6 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.broadcast %8 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %11 = tt.addptr %9, %10 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    tt.store %11, %12 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_s8_to_bf16_conversion
  tt.func @test_s8_to_bf16_conversion(%in: tensor<32xi8, #blocked>) {
    // We can't vectorize if we only process
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.sitofp
    // CHECK-NOT: llvm.sitofp
    %out = arith.sitofp %in : tensor<32xi8, #blocked> to tensor<32xbf16, #blocked>
    tt.return
  }
}

// -----
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_s8_to_bf16_vectorized_conversion
  tt.func @test_s8_to_bf16_vectorized_conversion(%in: tensor<16x16xi8, #mma>) {
    // CHECK-NOT: llvm.sitofp
    // 8 elements per thread => we should process 2 vectors of 4
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    // CHECK-NOT: llvm.inline_asm
    %out = arith.sitofp %in : tensor<16x16xi8, #mma> to tensor<16x16xbf16, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: sum_reduction
//       CHECK:  %[[M:.+]] = llvm.mlir.constant(-1 : i32) : i32
//       CHECK:   nvvm.redux.sync  add %{{.*}}, %[[M]]
//       CHECK:   nvvm.barrier0
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.barrier0
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sum_reduction(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1024> : tensor<1x1xi32, #blocked>
    %0 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<1x1xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<1x1x!tt.ptr<i32>, #blocked>, tensor<1x1xi32, #blocked>
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<1x1x!tt.ptr<i32>, #blocked> -> tensor<1x1024x!tt.ptr<i32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1x1024x!tt.ptr<i32>, #blocked>, tensor<1x1024xi32, #blocked>
    %10 = tt.load %9 : tensor<1x1024x!tt.ptr<i32>, #blocked>
    %11 = "tt.reduce"(%10) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %15 = arith.addi %arg2, %arg3 : i32
      tt.reduce.return %15 : i32
    }) : (tensor<1x1024xi32, #blocked>) -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = ttg.convert_layout %11 : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xi32, #blocked1>
    %13 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked1>
    %14 = tt.addptr %13, %0 : tensor<1x!tt.ptr<i32>, #blocked1>, tensor<1xi32, #blocked1>
    tt.store %14, %12 : tensor<1x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  // CHECK-LABEL: reduce_bools
  tt.func public @reduce_bools(%arg: tensor<256x2xi1, #blocked>) {
    // CHECK: llvm.mlir.addressof @global_smem
    %24 = "tt.reduce"(%arg) <{axis = 1 : i32}> ({
    ^bb0(%arg4: i1, %arg5: i1):
      %48 = arith.ori %arg4, %arg5 : i1
      tt.reduce.return %48 : i1
    }) : (tensor<256x2xi1, #blocked>) -> tensor<256xi1, #slice>
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: inline_asm
  tt.func public @inline_asm(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    %3 = tt.load %2 : tensor<512x!tt.ptr<i8>, #blocked>
// CHECK: %{{.*}} = llvm.inline_asm asm_dialect = att "shl.b32 $0, $0, 3;", "=r,r" %{{.*}} : (vector<4xi8>) -> vector<4xi8>
    %4 = tt.elementwise_inline_asm "shl.b32 $0, $0, 3;" {constraints = "=r,r", packed_element = 4 : i32, pure = true} %3 : tensor<512xi8, #blocked> -> tensor<512xi8, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    tt.store %6, %4 : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: inline_asm_pack_16bit
  tt.func public @inline_asm_pack_16bit(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    %3 = tt.load %2 : tensor<512x!tt.ptr<i8>, #blocked>
// CHECK: %{{.*}} = llvm.inline_asm asm_dialect = att "shl.b16 $0, $0, 3;", "=h,h" %{{.*}} : (vector<2xi8>) -> vector<2xi8>
    %4 = tt.elementwise_inline_asm "shl.b16 $0, $0, 3;" {constraints = "=h,h", packed_element = 2 : i32, pure = true} %3 : tensor<512xi8, #blocked> -> tensor<512xi8, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    tt.store %6, %4 : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_slice
//  CHECK-NOT: st.shared
//  CHECK-NOT: ld.shared
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 4, 2], warpsPerCTA = [2, 4, 2], order = [2, 0, 1], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#sliced2 = #ttg.slice<{dim = 2, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_slice() attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<4x1xi1, #sliced2>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.ori %arg0, %arg1 : i1
      tt.reduce.return %1 : i1
    }) : (tensor<4x1xi1, #sliced2>) -> tensor<4xi1, #ttg.slice<{dim = 1, parent = #sliced2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_md_slice
//  CHECK: st.shared
//  CHECK: st.shared
//  CHECK: ld.shared
//  CHECK: st.shared
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#sliced = #ttg.slice<{dim = 2, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_md_slice(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x128xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %18 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %18 : f32
    }) {allocation.offset = 0 : i32} : (tensor<2x128xf32, #sliced>) -> tensor<2xf32, #ttg.slice<{dim = 1, parent = #sliced}>>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 8, perPhase=1, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @i16_mma_layout(%f16_inp: tensor<16x16xf16, #blocked0>, %i16_inp: tensor<16x16xi16, #blocked0>) {
    // CHECK-LABEL: @i16_mma_layout

    %f16_shared = ttg.local_alloc %f16_inp : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %i16_shared = ttg.local_alloc %i16_inp : (tensor<16x16xi16, #blocked0>) -> !ttg.memdesc<16x16xi16, #shared0, #smem>

    // CHECK: nvgpu.ldmatrix
    // CHECK: nvgpu.ldmatrix

    %f16_dot = ttg.local_load %f16_shared : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    %i16_dot = ttg.local_load %i16_shared : !ttg.memdesc<16x16xi16, #shared0, #smem> -> tensor<16x16xi16, #dot_operand_b>

    // CHECK: llvm.sitofp %{{.*}} : i16 to f16

    %converted_i16 = arith.sitofp %i16_dot : tensor<16x16xi16, #dot_operand_b> to tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32

    %out = tt.dot %f16_dot, %converted_i16, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.return
  tt.func public @convert_single_element() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %0 = ttg.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element_and_add
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  tt.func public @convert_single_element_and_add() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %cst2 = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    %1 = arith.addf %0, %cst2 : tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_load
  // CHECK: llvm.load
  // CHECK-SAME: {alignment = 8 : i64} : !llvm.ptr<3> -> vector<8xi8>
  // CHECK-NOT: llvm.load
  tt.func public @vectorize_shmem_load(%shmem : !ttg.memdesc<16x16xi8, #shared, #smem>) {
    %0 = ttg.local_load %shmem : !ttg.memdesc<16x16xi8, #shared, #smem> -> tensor<16x16xi8, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_store
  // CHECK: llvm.store
  // CHECK-SAME: {alignment = 64 : i64} : vector<16xi32>, !llvm.ptr<3>
  // CHECK-NOT: llvm.store
  tt.func public @vectorize_shmem_store(%block : tensor<64x64xi32, #blocked>) {
    %0 = ttg.local_alloc %block : (tensor<64x64xi32, #blocked>) -> !ttg.memdesc<64x64xi32, #shared, #smem>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: abs_is_int_min_poison
  // CHECK: %{{.*}} = "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (i32) -> i32
  tt.func @abs_is_int_min_poison(%arg0 : tensor<256xi32, #blocked0>) {
    %abs = math.absi %arg0 : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_load_bf16
  // CHECK: llvm.extractelement {{.*}} : vector<8xbf16>
  tt.func public @test_local_load_bf16() {
    %c0_i32 = arith.constant 0 : i32
    %19 = ttg.local_alloc : () -> !ttg.memdesc<1x1x2048xbf16, #shared, #smem, mutable>
    %22 = ttg.memdesc_subview %19[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x1x2048xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x2048xbf16, #shared, #smem, mutable>
    %39 = ttg.local_load %22 : !ttg.memdesc<1x2048xbf16, #shared, #smem, mutable> -> tensor<1x2048xbf16, #blocked>
    %40 = arith.extf %39 : tensor<1x2048xbf16, #blocked> to tensor<1x2048xf32, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store
  // CHECK: llvm.store
  tt.func public @test_local_store(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<1xf32, #blocked> -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_st
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  tt.func public @tensor_memory_st(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tmem_store %cst_0, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store_subview
  // CHECK: llvm.store
  tt.func public @test_local_store_subview(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    %sv = ttg.memdesc_subview %0[%c0_i32] : !ttg.memdesc<1xf32, #shared, #smem, mutable> -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    ttg.local_store %arg0, %sv : tensor<1xf32, #blocked> -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: print_ptr
  // CHECK: llvm.call @vprintf(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  tt.func @print_ptr(%arg0 : tensor<256x!tt.ptr<i32>, #blocked0>) {
    tt.print "ptr: " {hex = false, isSigned = array<i32: 0>} : %arg0 : tensor<256x!tt.ptr<i32>, #blocked0>
    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Test that %u format specifier is used if isSigned is false
  // CHECK: llvm.mlir.global internal constant @printfFormat_0("{{.*}}int32 tensor: %u{{.*}}")
  // CHECK-LABEL: print_int32_tensor_issigned_off
  // CHECK: llvm.call @vprintf(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  tt.func @print_int32_tensor_issigned_off(%arg0 : i32) {
    tt.print "int32 tensor: " {hex = false, isSigned = array<i32: 0>} : %arg0 : i32
    tt.return
  }
}

// -----
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Test that %i format specifier is used if isSigned is true
  // CHECK: llvm.mlir.global internal constant @printfFormat_0("{{.*}}int32 tensor: %i{{.*}}")
  // CHECK-LABEL: print_int32_tensor_issigned_on
  // CHECK: llvm.call @vprintf(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  tt.func @print_int32_tensor_issigned_on(%arg0 : i32) {
    tt.print "int32 tensor: " {hex = false, isSigned = array<i32: 1>} : %arg0 : i32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @int32_to_bf16(%arg0: tensor<256xi32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: @int32_to_bf16
    // CHECK: llvm.sitofp %{{.*}} : i32 to bf16
    %a = arith.sitofp %arg0 : tensor<256xi32, #blocked> to tensor<256xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @bf16_to_int32(%arg0: tensor<256xbf16, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: @bf16_to_int32
    // CHECK: llvm.fptosi %{{.*}} : bf16 to i32
    %a = arith.fptosi %arg0 : tensor<256xbf16, #blocked> to tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: llvm.mlir.global internal constant @assertFunc_0("unknown\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global internal constant @assertFile_0("inner_call\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global internal constant @assertMessage_0("assert text\00") {addr_space = 0 : i32}
// CHECK: llvm.call @__assertfail
// CHECK: nvvm.barrier0
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: tensor<1xi1, #blocked>) {
    tt.assert %arg0, "assert text" : tensor<1xi1, #blocked> loc(#loc5)
    tt.return
  }
}
#loc1 = loc("outer_call":33:8)
#loc2 = loc("top_func":47:8)
#loc3 = loc("inner_call":29:28)
#loc4 = loc(callsite(#loc3 at #loc1))
#loc5 = loc(callsite(#loc4 at #loc2))

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @log1pf_scan(%39: tensor<32x16xf32, #blocked>) attributes {noinline = false} {
    // CHECK: log1pf_scan
    // non-speculatable ops will introduce a cond_br; extern_elementwise with pure = true should be considered speculatable.
    // CHECK-NOT: llvm.cond_br
    %40 = "tt.scan"(%39) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%arg5: f32, %arg6: f32):
      %43 = tt.extern_elementwise %arg5 {libname = "", libpath = "", pure = true, symbol = "__nv_log1pf"} : (f32) -> f32
      %44 = arith.addf %43, %43 : f32
      tt.scan.return %44 : f32
    }) : (tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK: inline_asm_pack
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // check specifically for the case where asm has two results, pack > 1, and the result bitwidth is < 32
  tt.func public @inline_asm_pack(%80: tensor<64x64xi8, #blocked>) attributes {noinline = false} {
    // CHECK: llvm.inline_asm asm_dialect {{.*}} (vector<4xi8>) -> !llvm.struct<(vector<2xbf16>, vector<2xbf16>, vector<2xbf16>, vector<2xbf16>)>
    %83:2 = tt.elementwise_inline_asm "" {constraints = "=r,=r,=r,=r,r", packed_element = 4 : i32, pure = true} %80 : tensor<64x64xi8, #blocked> -> tensor<64x64xbf16, #blocked>, tensor<64x64xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

tt.func @gather_in_shared(%arg0: tensor<16x4xi32, #blocked1>, %arg1: tensor<8x4xf32, #blocked>) {
  // CHECK-LABEL: gather_in_shared

  // CHECK: [[S0:%.*]] = llvm.extractvalue %arg1[0]

  // CHECK: [[SMEM_BASE:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM:%.*]] = llvm.getelementptr [[SMEM_BASE]]
  // CHECK: store [[S0]]
  // CHECK-NEXT: nvvm.barrier0

  // CHECK: [[I0:%.*]] = llvm.extractvalue %arg0[0]

  // CHECK: [[IDX:%.*]] = llvm.add {{.*}}, [[I0]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM]][[[IDX]]]
  // CHECK-NEXT: [[OUT0:%.*]] = llvm.load [[PTR]]

  // CHECK: insertvalue [[OUT0]], {{.*}}[0]

  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<8x4xf32, #blocked>, tensor<16x4xi32, #blocked1>) -> tensor<16x4xf32, #blocked1>
  tt.return
}

}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [1, 1]}>
#dot = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

tt.func @gather_in_shared_dot_input(%arg0: tensor<16x4xi32, #blocked>, %arg1: tensor<8x4xf32, #dot>) {
  // CHECK-LABEL: gather_in_shared_dot_input

  // CHECK: [[S0:%.*]] = llvm.extractvalue %arg1[0]
  // CHECK: [[S1:%.*]] = llvm.extractvalue %arg1[1]
  // CHECK: [[S2:%.*]] = llvm.extractvalue %arg1[2]
  // CHECK: [[S3:%.*]] = llvm.extractvalue %arg1[3]

  // CHECK: [[SMEM_BASE:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM:%.*]] = llvm.getelementptr [[SMEM_BASE]]
  // CHECK: store [[S0]]
  // CHECK: store [[S1]]
  // CHECK: store [[S2]]
  // CHECK: store [[S3]]
  // CHECK-NEXT: nvvm.barrier0

  // CHECK: [[I0:%.*]] = llvm.extractvalue %arg0[0]

  // CHECK: [[IDX:%.*]] = llvm.add {{.*}}, [[I0]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM]][[[IDX]]]
  // CHECK-NEXT: [[OUT0:%.*]] = llvm.load [[PTR]]

  // CHECK: insertvalue [[OUT0]], {{.*}}[0]

  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<8x4xf32, #dot>, tensor<16x4xi32, #blocked>) -> tensor<16x4xf32, #blocked>
  tt.return
}

}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

  tt.func public @ampere_s8_to_fp16_conversion_opIdx1(%1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: ampere_s8_to_fp16_conversion_opIdx1
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> to tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
}

}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx0(%1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: @ampere_s8_to_fp16_conversion_opIdx0
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0 , parent = #mma, kWidth = 4}>> to tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [4, 4, 2], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 0], [0, 0], [0, 0], [0, 0]], lane = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: expand_dims_linear_layout
tt.func private @expand_dims_linear_layout() -> tensor<1x4xi32, #linear> {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x4xi32, #linear>
  // CHECK: return %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  tt.return %1 : tensor<1x4xi32, #linear>
}

// CHECK-LABEL: reshape_linear_layout_broadcasting
tt.func private @reshape_linear_layout_broadcasting(%arg0: tensor<32x4xbf16, #linear>) -> tensor<32x4x1xbf16, #blocked> {
  // CHECK-COUNT-16: extractvalue
  // CHECK-COUNT-16: insertvalue
  %0 = tt.reshape %arg0 : tensor<32x4xbf16, #linear> -> tensor<32x4x1xbf16, #blocked>
  tt.return %0 : tensor<32x4x1xbf16, #blocked>
}

}


// -----

#linear1 = #ttg.linear<{register = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0]], lane = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], warp = [[4, 0, 0, 0], [8, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: split_linear
tt.func @split_linear(%arg : tensor<128x2x2x2xf32, #linear1>) {
  // CHECK: %[[E0:.+]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[E1:.+]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[E2:.+]] = llvm.extractvalue %{{.*}}[2]
  // CHECK: %[[E3:.+]] = llvm.extractvalue %{{.*}}[3]
  // CHECK: llvm.insertvalue %[[E0]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[E2]], %{{.*}}[1]
  // CHECK: llvm.insertvalue %[[E1]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[E3]], %{{.*}}[1]
  %outLHS, %outRHS = tt.split %arg : tensor<128x2x2x2xf32, #linear1> -> tensor<128x2x2xf32, #linear2>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: split_stride
  tt.func public @split_stride(%arg0: tensor<128x64x2xf32, #blocked>) {
  // CHECK: %[[E0:.+]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[E1:.+]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[E64:.+]] = llvm.extractvalue %{{.*}}[64]
  // CHECK: %[[E65:.+]] = llvm.extractvalue %{{.*}}[65]
  // CHECK: llvm.insertvalue %[[E0]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[E1]], %{{.*}}[1]
  // CHECK: llvm.insertvalue %[[E64]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[E65]], %{{.*}}[1]
    %outLHS, %outRHS = tt.split %arg0 : tensor<128x64x2xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: join_stride
  tt.func public @join_stride(%arg0: tensor<128x64xf32, #blocked1>, %arg1: tensor<128x64xf32, #blocked1>) {
  // CHECK: %[[A0:.+]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[A1:.+]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[B0:.+]] = llvm.extractvalue %{{.*}}[0]
  // CHECK: %[[B1:.+]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: llvm.insertvalue %[[A0]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[A1]], %{{.*}}[1]
  // CHECK: llvm.insertvalue %[[B0]], %{{.*}}[64]
  // CHECK: llvm.insertvalue %[[B1]], %{{.*}}[65]
    %r = tt.join %arg0, %arg1 : tensor<128x64xf32, #blocked1> -> tensor<128x64x2xf32, #blocked>
    tt.return
  }
}
