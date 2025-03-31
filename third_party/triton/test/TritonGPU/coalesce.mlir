// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice1dim1 = #ttg.slice<{dim = 1, parent = #blocked1}>
#slice2dim0 = #ttg.slice<{dim = 0, parent = #blocked2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: [[row_layout:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: [[load_ptr:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
// CHECK: [[load_mask:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64xi1, [[row_layout]]>
// CHECK: [[load_other:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64xf32, [[row_layout]]>
// CHECK: [[load_val:%.*]] = tt.load [[load_ptr]], [[load_mask]], [[load_other]] : tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
// CHECK: [[store_ptr:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[col_layout]]>
// CHECK: [[store_val:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64xf32, [[col_layout]]>
// CHECK: [[store_mask:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64xi1, [[col_layout]]>
// CHECK: tt.store [[store_ptr]], [[store_val]], [[store_mask]]
tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg1: i32 {tt.divisibility = 16 : i32},
                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg3: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %00 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1dim1>
  %01 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice2dim0>
  %1 = tt.expand_dims %00 {axis = 1 : i32} : tensor<64xi32, #slice1dim1> -> tensor<64x1xi32, #blocked1>
  %2 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked1>
  %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %6 = tt.expand_dims %01 {axis = 0 : i32} : tensor<64xi32, #slice2dim0> -> tensor<1x64xi32, #blocked2>
  %7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %8 = tt.broadcast %6 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %9 = ttg.convert_layout %8 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %12 = tt.addptr %11, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %13 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked2>
  %14 = arith.muli %6, %13 : tensor<1x64xi32, #blocked2>
  %15 = tt.broadcast %12 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %16 = tt.broadcast %14 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %17 = ttg.convert_layout %16 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %18 = tt.addptr %15, %17 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %19 = tt.load %10, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.store %18, %19, %cst : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {


// CHECK: [[NARROW_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[WIDE_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi "slt", %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[WIDE_LAYOUT]]>
    tt.store %16, %14, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-NOT: sizePerThread = [4]
// CHECK: #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-NOT: sizePerThread = [4]
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi "slt", %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %17 = arith.truncf %14 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    tt.store %16, %17, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
}

}

// -----

// COM: Reproducer for issue #3866
// CHECK-LABEL: @test_3866
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<64x16xf16>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @test_3866(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i64) {
    %0 = tt.make_tensor_ptr %arg0, [%arg2, %arg2], [%arg2, %arg2], [%arg1, %arg1] {order = array<i32: 1, 0>} : <tensor<64x16xf16>>
    %1 = tt.load %0 : !tt.ptr<tensor<64x16xf16>>
    tt.return
  }
}

// -----

// COM: Reproducer for issue #5122
// CHECK-LABEL: @test_5122
module {
  tt.func public @test_5122(%arg0: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi sgt, %arg0, %c1_i32 : i32
    scf.if %0 {
      %1 = scf.if %0 -> (i32) {
        scf.yield %c1_i32 : i32
      } else {
        scf.yield %c1_i32 : i32
      }
      %2 = arith.cmpi sgt, %1, %c1_i32 : i32
      %3 = scf.if %2 -> (i32) {
        scf.yield %c1_i32 : i32
      } else {
        scf.yield %c1_i32 : i32
      }
      %4 = scf.for %arg1 = %1 to %1 step %c1_i32 iter_args(%arg2 = %3) -> (i32) : i32 {
        %5 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %5 : i32
      }
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK: [[COALESCED_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: @coalesce_poison
tt.func @coalesce_poison(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
  %2 = ttg.convert_layout %1 : tensor<128xi32, #blocked1> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
  %4 = ttg.convert_layout %3 : tensor<128x1xi32, #blocked2> -> tensor<128x1xi32, #blocked3>
  %5 = tt.broadcast %4 {axis = 1 : i32} : tensor<128x1xi32, #blocked3> -> tensor<128x64xi32, #blocked3>
  %6 = ttg.convert_layout %5 : tensor<128x64xi32, #blocked3> -> tensor<128x64xi32, #blocked>
  %7 = tt.addptr %0, %6 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>

  %8 = ub.poison : tensor<128x64x!tt.ptr<f16>, #blocked>
  // CHECK: scf.if
  %9 = scf.if %arg2 -> (tensor<128x64x!tt.ptr<f16>, #blocked>) {
    scf.yield %8 : tensor<128x64x!tt.ptr<f16>, #blocked>
  } else {
    scf.yield %7 : tensor<128x64x!tt.ptr<f16>, #blocked>
  }
  // CHECK: [[PTR:%.*]] = ttg.convert_layout %{{.*}} : tensor<128x64x!tt.ptr<f16>, #{{.*}}> -> tensor<128x64x!tt.ptr<f16>, [[COALESCED_LAYOUT]]>
  // CHECK-NEXT: tt.load [[PTR]]
  %10 = tt.load %9 : tensor<128x64x!tt.ptr<f16>, #blocked>
  tt.return
}

}
