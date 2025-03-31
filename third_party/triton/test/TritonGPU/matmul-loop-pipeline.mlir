// RUN: triton-opt %s -split-input-file -tritongpu-pipeline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @softmax_kernel
tt.func public @softmax_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked>
  %0 = tt.get_program_id x : i32
  %1 = tt.get_num_programs x : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
  %3 = tt.splat %arg5 : i32 -> tensor<128xi32, #blocked>
  // CHECK: [[MASK:%.*]] = arith.cmpi slt, {{.*}} tensor<128xi32,
  %4 = arith.cmpi slt, %2, %3 : tensor<128xi32, #blocked>
  // CHECK: scf.for
  scf.for %arg6 = %0 to %arg4 step %1  : i32 {
    %5 = tt.splat %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %2 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    // CHECK: [[RESULT:%.*]] = ttg.local_load
    // CHECK-NEXT: arith.select [[MASK]], [[RESULT]], %cst
    %7 = tt.load %6, %4, %cst {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %2 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %9, %7, %4 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked>
  } {tt.num_stages = 2 : i32}
  tt.return
}

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @scalar_load
tt.func public @scalar_load(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: f32) -> f32 {
  %c1_i32 = arith.constant 1 : i32
  %2 = scf.for %i = %arg1 to %arg2 step %c1_i32 iter_args(%k = %arg3) -> f32 : i32 {
    // CHECK: tt.load %arg0
    %0 = tt.load %arg0 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.ptr<f32>
    %1 = arith.addf %0, %k {loop.cluster = 1 : i32, loop.stage = 0 : i32} : f32
    %2 = arith.addf %1, %k {loop.cluster = 0 : i32, loop.stage = 1 : i32} : f32
    scf.yield %2 : f32
  } {num_stages = 2 : i32}
  tt.return %2 : f32
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @make_tensor_desc_epilogue
tt.func public @make_tensor_desc_epilogue(%arg0: i32, %arg1: !tt.ptr<f32>, %arg2: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  // CHECK: scf.for
  scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 : i32 {
    %1 = tt.splat %arg1 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %2 = tt.load %1 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x256x!tt.ptr<f32>, #blocked>
    %3 = arith.addf %2, %2 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : tensor<128x256xf32, #blocked>
    %4 = arith.cmpi eq, %arg3, %c1_i32 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : i32
    // CHECK: scf.if
    scf.if %4 {
      // CHECK-NOT: tt.make_tensor_descriptor
      // CHECK: tt.experimental_tensormap_create
      // CHECK-NEXT: tt.experimental_tensormap_fenceproxy_acquire
      %5 = tt.make_tensor_descriptor %arg1, [%arg2, %arg2], [%c1_i64, %c1_i64] : <f32>, <tensor<128x256xf32, #nvmma_128>>
    } {loop.cluster = 5 : i32, loop.stage = 2 : i32}
  } {tt.num_stages = 3 : i32}
  tt.return
}

}
