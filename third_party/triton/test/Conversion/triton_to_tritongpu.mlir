// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=2' | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @ops() {
  // CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {{.*}}
  %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
  %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
  %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @load_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if LoadOp is lowered properly (see #771)
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<128xi1>
  %other = arith.constant dense<0.0e+0> : tensor<128xf32>
  // CHECK: %{{.*}} = tt.load %{{.*}} : {{.*}}
  %a = tt.load %ptrs : tensor<128x!tt.ptr<f32>>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}} : {{.*}}
  %b = tt.load %ptrs, %mask : tensor<128x!tt.ptr<f32>>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}}, %{{.*}} : {{.*}}
  %c = tt.load %ptrs, %mask, %other : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %a : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %b : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %c : tensor<128x!tt.ptr<f32>>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @reduce_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if the total number of threadsPerWarp is 32
  // Test if the total number of warps is 2
  // CHECK: #[[blocked0:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: #[[blocked1:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: #[[blocked2:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {{.*}}
  %c0 = arith.constant dense<1.00e+00> : tensor<4x4xf32>
  %c1 = arith.constant dense<2.00e+00> : tensor<8x2xf32>
  %c2 = arith.constant dense<3.00e+00> : tensor<16x16xf32>
  // CHECK: (tensor<4x4xf32, #[[blocked0]]>) -> tensor<4xf32, #ttg.slice<{dim = 0, parent = #[[blocked0]]}>>
  %c0_ = "tt.reduce" (%c0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %add = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<4x4xf32>) -> tensor<4xf32>
  // CHECK: (tensor<8x2xf32, #[[blocked1]]>) -> tensor<2xf32, #ttg.slice<{dim = 0, parent = #[[blocked1]]}>
  %c1_ = "tt.reduce" (%c1) ({
  ^bb0(%arg3: f32, %arg4: f32):
    %add = arith.addf %arg3, %arg4 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<8x2xf32>) -> tensor<2xf32>
  // CHECK: (tensor<8x2xf32, #[[blocked1]]>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #[[blocked1]]}>>
  %c2_ = "tt.reduce" (%c1) ({
  ^bb0(%arg5: f32, %arg6: f32):
    %add = arith.addf %arg5, %arg6 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<8x2xf32>) -> tensor<8xf32>
  // CHECK: (tensor<16x16xf32, #[[blocked2]]>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[blocked2]]}>>
  %c3_ = "tt.reduce" (%c2) ({
  ^bb0(%arg7: f32, %arg8: f32):
    %add = arith.addf %arg7, %arg8 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>

  tt.return
}
}


// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func public @select_op(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i1) attributes {noinline = false} {
  // CHECK-LABEL: select_op
  %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>

  // CHECK: %{{.*}} = arith.select %arg2, %{{.*}}, %{{.*}} : tensor<128xf32, #blocked>
  %4 = arith.select %arg2, %cst, %3 : tensor<128xf32>

  %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.store %6, %4 : tensor<128x!tt.ptr<f32>>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @arith_splat_bool(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK-LABEL: arith_splat_bool

  // Test arith.constant with splatted bool.
  // CHECK-NEXT: arith.constant dense<true> : tensor<128xi1, #{{.*}}>
  %mask = arith.constant dense<true> : tensor<128xi1>
  tt.return
}
}

// -----

// CHECK-LABEL: gather_op
tt.func @gather_op() {
  %cst = arith.constant dense<1.0> : tensor<128x4xf32>
  %cst_0 = arith.constant dense<1> : tensor<256x4xi32>
  // CHECK: tt.gather %{{.*}}[%{{.*}}] {axis = 0 : i32} : (tensor<128x4xf32, #blocked>, tensor<256x4xi32, #blocked>) -> tensor<256x4xf32, #blocked>
  %0 = tt.gather %cst[%cst_0] {axis = 0 : i32} : (tensor<128x4xf32>, tensor<256x4xi32>) -> tensor<256x4xf32>
  tt.return
}

// -----

// CHECK: [[SLICE_PARENT:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>

// CHECK: @gather4_layout
tt.func @gather4_layout(%arg0: !tt.tensordesc<tensor<1x128xf32>>, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<1> : tensor<32xi32>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  %0 = tt.descriptor_gather %arg0[%cst, %arg1] : (!tt.tensordesc<tensor<1x128xf32>>, tensor<32xi32>, i32) -> tensor<32x128xf32>
  %1 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>>
  tt.store %1, %0 : tensor<32x128x!tt.ptr<f32>>
  tt.return
}

// CHECK: @scatter4_layout
tt.func @scatter4_layout(%arg0: !tt.tensordesc<tensor<1x128xf32>>, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<1> : tensor<32xi32>
  %0 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>>
  %1 = tt.load %0 : tensor<32x128x!tt.ptr<f32>>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  tt.descriptor_scatter %arg0[%cst, %arg1], %1 : !tt.tensordesc<tensor<1x128xf32>>, tensor<32xi32>, i32, tensor<32x128xf32>
  tt.return
}

// -----

// CHECK-LABEL: @ub_poison
tt.func @ub_poison() {
  // CHECK-NEXT: ub.poison : tensor<128x64xf16, #blocked>
  %0 = ub.poison : tensor<128x64xf16>
  tt.return
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @partition_axis_info
tt.func @partition_axis_info(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  partition0(%arg2: !tt.ptr<i32>) num_warps(2) {
    %splatted = tt.splat %arg2 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked2>
    %input = tt.load %splatted : tensor<256x!tt.ptr<i32>, #blocked2>
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  tt.return
}

}
