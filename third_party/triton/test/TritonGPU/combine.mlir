// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-remove-layout-conversions -cse 2>&1 | FileCheck %s

#layout0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

#layout2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#layout3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>


module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK: [[$target_layout:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: cst
tt.func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = ttg.convert_layout %cst : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return %cst : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: range
tt.func @range() -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = ttg.convert_layout %0 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: splat
tt.func @splat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.splat %arg0 : i32 -> tensor<1024xi32, #layout0>
  %1 = ttg.convert_layout %0 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: remat
tt.func @remat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %2 = arith.muli %0, %1 : tensor<1024xi32, #layout0>
  %3 = ttg.convert_layout %2 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  %4 = tt.splat %arg0 : i32 -> tensor<1024xi32, #layout0>
  %5 = ttg.convert_layout %2 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  %6 = arith.addi %3, %5 : tensor<1024xi32, #layout1>
  tt.return %6: tensor<1024xi32, #layout1>
  // CHECK: %[[A:.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %[[C:.+]] = arith.muli %[[A]], %[[A]] : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %[[D:.+]] = arith.addi %[[C]], %[[C]] : tensor<1024xi32, [[$target_layout]]>
  // CHECK: tt.return %[[D]] : tensor<1024xi32, [[$target_layout]]>
}

// Always rematerialize single value loads
// CHECK-LABEL: remat_single_value
tt.func @remat_single_value(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #layout1>
  %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #layout1>
  // CHECK-NOT: ttg.convert_layout
  %2 = ttg.convert_layout %1 : tensor<1xi32, #layout1> -> tensor<1xi32, #layout0>
  %3 = ttg.convert_layout %0 : tensor<1x!tt.ptr<i32>, #layout1> -> tensor<1x!tt.ptr<i32>, #layout0>
  tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #layout0>
  tt.return
}

// CHECK-LABEL: remat_fast_load
tt.func @remat_fast_load(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #layout1>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #layout1>
  %2 = tt.addptr %0, %1 : tensor<16x!tt.ptr<i32>, #layout1>, tensor<16xi32, #layout1>
  %3 = tt.load %2 : tensor<16x!tt.ptr<i32>, #layout1>
  // CHECK-NOT: ttg.convert_layout
  %4 = ttg.convert_layout %3 : tensor<16xi32, #layout1> -> tensor<16xi32, #layout0>
  %5 = ttg.convert_layout %2 : tensor<16x!tt.ptr<i32>, #layout1> -> tensor<16x!tt.ptr<i32>, #layout0>
  tt.store %5, %4 : tensor<16x!tt.ptr<i32>, #layout0>
  tt.return
}

// Hoist the convert on top of ext to make it cheaper.
// CHECK-LABEL: hoist_above_ext
tt.func @hoist_above_ext(%arg0: tensor<1024xf16, #layout0>, %arg1: f32) -> tensor<1024xf32, #layout1> {
// CHECK: %[[CVT:.+]] = ttg.convert_layout
// CHECK: arith.extf %[[CVT]]
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return
  %0 = arith.extf %arg0 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %1 = tt.splat %arg1 : f32 -> tensor<1024xf32, #layout0>
  %2 = arith.addf %0, %1 : tensor<1024xf32, #layout0>
  %3 = ttg.convert_layout %2 : tensor<1024xf32, #layout0> -> tensor<1024xf32, #layout1>
  tt.return %3 : tensor<1024xf32, #layout1>
}

// CHECK-LABEL: hoist_above_ext2
tt.func @hoist_above_ext2(%arg0: tensor<1024xf16, #layout0>, %arg1: f16) -> tensor<1024xf32, #layout1> {
// CHECK: %[[CVT:.+]] = ttg.convert_layout
// CHECK: arith.extf %[[CVT]]
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return
  %0 = arith.extf %arg0 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %1 = tt.splat %arg1 : f16 -> tensor<1024xf16, #layout0>
  %2 = arith.extf %1 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %3 = arith.addf %0, %2 : tensor<1024xf32, #layout0>
  %4 = ttg.convert_layout %3 : tensor<1024xf32, #layout0> -> tensor<1024xf32, #layout1>
  tt.return %4 : tensor<1024xf32, #layout1>
}

/// CHECK-LABEL: hoist_above_fptofp
tt.func @hoist_above_fptofp(%arg0: tensor<1024xf8E4M3FNUZ, #layout0>) -> tensor<1024xf32, #layout1> {
// CHECK: %[[CVT:.+]] = ttg.convert_layout
// CHECK: tt.fp_to_fp %[[CVT]]
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return
  %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<1024xf8E4M3FNUZ, #layout0> -> tensor<1024xf32, #layout0>
  %1 = ttg.convert_layout %0 : tensor<1024xf32, #layout0> -> tensor<1024xf32, #layout1>
  tt.return %1 : tensor<1024xf32, #layout1>
}

/// CHECK-LABEL: dont_hoist_above_trunc_fptofp
tt.func @dont_hoist_above_trunc_fptofp(%arg0: tensor<1024xf32, #layout0>) -> tensor<1024xf8E4M3FNUZ, #layout1> {
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[FP8:.+]] = tt.fp_to_fp
// CHECK: ttg.convert_layout %[[FP8]]
// CHECK: tt.return
  %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<1024xf32, #layout0> -> tensor<1024xf8E4M3FNUZ, #layout0>
  %1 = ttg.convert_layout %0 : tensor<1024xf8E4M3FNUZ, #layout0> -> tensor<1024xf8E4M3FNUZ, #layout1>
  tt.return %1 : tensor<1024xf8E4M3FNUZ, #layout1>
}

// Hoist the convert on top of broadcast to make it cheaper.
// CHECK-LABEL: hoist_above_broadcast
tt.func @hoist_above_broadcast(%arg0: tensor<1024x1xf32, #layout2>, %arg1: f32) -> tensor<1024x128xf32, #layout3> {
// CHECK: %[[CVT:.+]] = ttg.convert_layout
// CHECK: tt.broadcast %[[CVT]]
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return
  %0 = tt.broadcast %arg0 : tensor<1024x1xf32, #layout2> -> tensor<1024x128xf32, #layout2>
  %1 = tt.splat %arg1 : f32 -> tensor<1024x128xf32, #layout2>
  %2 = arith.addf %0, %1 : tensor<1024x128xf32, #layout2>
  %3 = ttg.convert_layout %2 : tensor<1024x128xf32, #layout2> -> tensor<1024x128xf32, #layout3>
  tt.return %3 : tensor<1024x128xf32, #layout3>
}


// CHECK-LABEL: if
tt.func @if(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: ttg.convert_layout
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout1>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : i32 -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout1>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout1>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #layout0>
  scf.if %4 {
    %6 = ttg.convert_layout %2 : tensor<1024xi32, #layout1> -> tensor<1024xi32, #layout0>
    tt.store %5, %6 : tensor<1024x!tt.ptr<i32>, #layout0>
  }
  tt.return
}

// CHECK-LABEL: if_convert_else_not
tt.func @if_convert_else_not(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : i32 -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : i32 -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = ttg.convert_layout %2 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    scf.yield %9 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: ttg.convert_layout
  tt.store %5, %8 : tensor<1024x!tt.ptr<i32>, #layout1>
  tt.return
}

// CHECK-LABEL: if_not_else_convert
tt.func @if_not_else_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : i32 -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : i32 -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    scf.yield %9 : tensor<1024xi32, #layout1>
  } else {
    %7 = ttg.convert_layout %3 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: ttg.convert_layout
  tt.store %5, %8 : tensor<1024x!tt.ptr<i32>, #layout1>
  tt.return
}

// CHECK-LABEL: if_else_both_convert
tt.func @if_else_both_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : i32 -> tensor<1024xi32, #layout0>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = ttg.convert_layout %2 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    %7 = ttg.convert_layout %3 : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // TODO(csigg): seems like the whole function is converted to layout1.
  // disabledCHECK: ttg.convert_layout
  // CHECK-NOT: ttg.convert_layout
  tt.store %5, %8 : tensor<1024x!tt.ptr<i32>, #layout1>
  tt.return
}

}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked0a = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1dim1 = #ttg.slice<{dim = 1, parent = #blocked1}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#slice2dim0 = #ttg.slice<{dim = 0, parent = #blocked2}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-DAG: [[$row_layout:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK-DAG: [[$col_layout:#.*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: [[$col_layout_novec:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

// CHECK-LABEL: @transpose
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: ttg.convert_layout
  // CHECK: [[loaded_val:%.*]] = tt.load {{.*}}, {{%cst.*}}, {{%cst.*}} : tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>
  // CHECK: [[cvt_val:%.*]] = ttg.convert_layout [[loaded_val]] : tensor<64x64xf32, [[$row_layout]]> -> tensor<64x64xf32, [[$col_layout]]>
  // CHECK: tt.store {{.*}}, [[cvt_val]], {{%cst.*}} : tensor<64x64x!tt.ptr<f32>, [[$col_layout]]>
  // CHECK: tt.return
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
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
  %19 = ttg.convert_layout %10 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
  %20 = ttg.convert_layout %cst_0 : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked3>
  %21 = ttg.convert_layout %cst : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
  %22 = tt.load %19, %20, %21 : tensor<64x64x!tt.ptr<f32>, #blocked3>
  %23 = ttg.convert_layout %22 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
  %24 = ttg.convert_layout %18 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked4>
  %25 = ttg.convert_layout %23 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked4>
  %26 = ttg.convert_layout %cst_0 : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked4>
  tt.store %24, %25, %26 : tensor<64x64x!tt.ptr<f32>, #blocked4>
  tt.return
}
}

// CHECK-LABEL: loop
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
  // CHECK-NOT: ttg.convert_layout
  // CHECK: [[loop_ret:%.*]]:2 = scf.for {{.*}} -> (tensor<64x64xf32, [[$row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>)
  // CHECK-NEXT: {{.*}} = tt.load {{.*}} : tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>
  // CHECK-NEXT: {{.*}} = arith.addf {{.*}} : tensor<64x64xf32, [[$row_layout]]>
  // CHECK-NEXT: {{.*}} = tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>, tensor<64x64xi32, [[$row_layout]]>
  // CHECK-NEXT: scf.yield {{.*}} : tensor<64x64xf32, [[$row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>
  // CHECK-NEXT: }
  // CHECK-NOT: ttg.convert_layout
  //     CHECK: {{.*}} = ttg.convert_layout [[loop_ret]]#0 : tensor<64x64xf32, [[$row_layout]]> -> tensor<64x64xf32, [[$col_layout_novec]]>
  // CHECK-NOT: ttg.convert_layout
  //    CHECK:  tt.return
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<64> : tensor<64x64xi32, #blocked1>
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
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
  %11:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %10) -> (tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>) {
    %23 = ttg.convert_layout %arg7 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %24 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked3>
    %25 = ttg.convert_layout %cst_1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
    %26 = tt.load %23, %24, %25 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %27 = ttg.convert_layout %26 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
    %28 = arith.addf %arg6, %27 : tensor<64x64xf32, #blocked1>
    %29 = tt.addptr %arg7, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    scf.yield %28, %29 : tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>
  }
  %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %13 = tt.addptr %12, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %14 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked2>
  %15 = arith.muli %6, %14 : tensor<1x64xi32, #blocked2>
  %16 = tt.broadcast %13 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %17 = tt.broadcast %15 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %18 = ttg.convert_layout %17 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %20 = ttg.convert_layout %19 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %21 = ttg.convert_layout %11#0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
  %22 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
  tt.store %20, %21, %22 : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.return
}
}

// CHECK-LABEL: loop_if
// CHECK-NOT: ttg.convert_layout
//     CHECK: scf.for
// CHECK-NOT: ttg.convert_layout
//     CHECK:   scf.if
// CHECK-NOT: ttg.convert_layout
//     CHECK:     scf.yield
//     CHECK:   else
//     CHECK:     scf.yield
// CHECK-NOT: ttg.convert_layout
//     CHECK:   scf.yield
//     CHECK: ttg.convert_layout
// CHECK-NOT: ttg.convert_layout
//     CHECK: tt.store
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @loop_if(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<64> : tensor<64x64xi32, #blocked1>
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
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
  %11:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %10) -> (tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>) {
    %33 = arith.cmpi "sgt", %arg5, %c0 : index
    %34 = scf.if %33 -> (tensor<64x64xf32, #blocked1>) {
      %23 = ttg.convert_layout %arg7 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
      %24 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked3>
      %25 = ttg.convert_layout %cst_1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
      %26 = tt.load %23, %24, %25 : tensor<64x64x!tt.ptr<f32>, #blocked3>
      %27 = ttg.convert_layout %26 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      scf.yield %27 : tensor<64x64xf32, #blocked1>
    } else {
      scf.yield %arg6 : tensor<64x64xf32, #blocked1>
    }
    %28 = arith.addf %arg6, %34 : tensor<64x64xf32, #blocked1>
    %29 = tt.addptr %arg7, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    scf.yield %28, %29 : tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>
  }
  %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %13 = tt.addptr %12, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %14 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked2>
  %15 = arith.muli %6, %14 : tensor<1x64xi32, #blocked2>
  %16 = tt.broadcast %13 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %17 = tt.broadcast %15 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %18 = ttg.convert_layout %17 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %20 = ttg.convert_layout %19 : tensor<64x64x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %21 = ttg.convert_layout %11#0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
  %22 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
  tt.store %20, %21, %22 : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.return
}
}

// CHECK-LABEL: vecadd
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @vecadd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
  // CHECK-NOT: ttg.convert_layout
  %c256_i32 = arith.constant 256 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c256_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<256xi32, #blocked5>
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked5>
  %4 = tt.splat %1 : i32 -> tensor<256xi32, #blocked5>
  %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked5>
  %6 = tt.splat %1 : i32 -> tensor<256xi32, #blocked5>
  %7 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked5>
  %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked5>
  %9 = arith.addi %6, %7 : tensor<256xi32, #blocked5>
  %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked5>
  %11 = arith.addi %4, %5 : tensor<256xi32, #blocked5>
  %12 = tt.addptr %8, %9 : tensor<256x!tt.ptr<f32>, #blocked5>, tensor<256xi32, #blocked5>
  %13 = tt.load %12 : tensor<256x!tt.ptr<f32>, #blocked5>
  %14 = ttg.convert_layout %13 : tensor<256xf32, #blocked5> -> tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
  %15 = tt.addptr %10, %11 : tensor<256x!tt.ptr<f32>, #blocked5>, tensor<256xi32, #blocked5>
  %16 = tt.load %15 : tensor<256x!tt.ptr<f32>, #blocked5>
  %17 = ttg.convert_layout %16 : tensor<256xf32, #blocked5> -> tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
  %18 = arith.addf %14, %17 : tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
  %19 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked5>
  %20 = arith.addi %2, %3 : tensor<256xi32, #blocked5>
  %21 = tt.addptr %19, %20 : tensor<256x!tt.ptr<f32>, #blocked5>, tensor<256xi32, #blocked5>
  %22 = ttg.convert_layout %18 : tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xf32, #blocked5>
  tt.store %21, %22 : tensor<256x!tt.ptr<f32>, #blocked5>
  tt.return
}
}

// Select has args with different element types
// CHECK-LABEL: select
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @select(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: ttg.convert_layout
  %cst = arith.constant dense<30000> : tensor<1x1xi32, #blocked2>
  %cst_0 = arith.constant dense<30000> : tensor<1x512xi32, #blocked2>
  %c512 = arith.constant 512 : i32
  %c30000 = arith.constant 30000 : i32
  %c0 = arith.constant 0 : i32
  %cst_1 = arith.constant dense<2048> : tensor<1x1xi32, #blocked2>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x512xf64, #blocked2>
  %0 = tt.get_program_id x : i32
  %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked0>
  %2 = ttg.convert_layout %1 : tensor<1xi32, #blocked0> -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x1xi32, #blocked1>
  %4 = ttg.convert_layout %3 : tensor<1x1xi32, #blocked1> -> tensor<1x1xi32, #blocked2>
  %5 = tt.splat %0 : i32 -> tensor<1x1xi32, #blocked2>
  %6 = arith.addi %5, %4 : tensor<1x1xi32, #blocked2>
  %7 = arith.cmpi "slt", %6, %cst_1 : tensor<1x1xi32, #blocked2>
  %8 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked0>
  %9 = ttg.convert_layout %8 : tensor<512xi32, #blocked0> -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x512xi32, #blocked2>
  %11 = arith.muli %6, %cst : tensor<1x1xi32, #blocked2>
  %12 = tt.broadcast %11 : tensor<1x1xi32, #blocked2> -> tensor<1x512xi32, #blocked2>
  %13 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<1x512x!tt.ptr<f64>, #blocked2>
  %14 = tt.broadcast %7 : tensor<1x1xi1, #blocked2> -> tensor<1x512xi1, #blocked2>
  %15 = scf.for %arg3 = %c0 to %c30000 step %c512 iter_args(%arg4 = %cst_2) -> (tensor<1x512xf64, #blocked2>) : i32 {
    %17 = tt.splat %arg3 : i32 -> tensor<1x512xi32, #blocked2>
    %18 = arith.addi %17, %10 : tensor<1x512xi32, #blocked2>
    %19 = arith.cmpi "slt", %18, %cst_0 : tensor<1x512xi32, #blocked2>
    %20 = arith.addi %18, %12 : tensor<1x512xi32, #blocked2>
    %21 = tt.addptr %13, %20 : tensor<1x512x!tt.ptr<f64>, #blocked2>, tensor<1x512xi32, #blocked2>
    %22 = arith.andi %19, %14 : tensor<1x512xi1, #blocked2>
    %23 = ttg.convert_layout %21 : tensor<1x512x!tt.ptr<f64>, #blocked2> -> tensor<1x512x!tt.ptr<f64>, #blocked3>
    %24 = ttg.convert_layout %22 : tensor<1x512xi1, #blocked2> -> tensor<1x512xi1, #blocked3>
    %25 = tt.load %23, %24 : tensor<1x512x!tt.ptr<f64>, #blocked3>
    %26 = ttg.convert_layout %25 : tensor<1x512xf64, #blocked3> -> tensor<1x512xf64, #blocked2>
    %27 = arith.andi %14, %19 : tensor<1x512xi1, #blocked2>
    %28 = arith.cmpf "olt", %arg4, %26 : tensor<1x512xf64, #blocked2>
    %29 = arith.andi %27, %28 : tensor<1x512xi1, #blocked2>
    %30 = arith.select %29, %26, %arg4 : tensor<1x512xi1, #blocked2>, tensor<1x512xf64, #blocked2>
    %31 = ttg.convert_layout %21 : tensor<1x512x!tt.ptr<f64>, #blocked2> -> tensor<1x512x!tt.ptr<f64>, #blocked3>
    %32 = ttg.convert_layout %30 : tensor<1x512xf64, #blocked2> -> tensor<1x512xf64, #blocked3>
    %33 = ttg.convert_layout %27 : tensor<1x512xi1, #blocked2> -> tensor<1x512xi1, #blocked3>
    tt.store %31, %32, %33 : tensor<1x512x!tt.ptr<f64>, #blocked3>
    scf.yield %30 : tensor<1x512xf64, #blocked2>
  }
  tt.return
}
}

// Make sure the following IR doesn't hang the compiler.
// CHECK-LABEL: long_func
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func public @long_func(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32, #blocked0>
  %cst_0 = arith.constant dense<5.000000e-04> : tensor<1024xf32, #blocked0>
  %cst_1 = arith.constant dense<0.999499976> : tensor<1024xf32, #blocked0>
  %cst_2 = arith.constant dense<1.000000e+04> : tensor<1024xf32, #blocked0>
  %cst_3 = arith.constant dense<5000> : tensor<1024xi32, #blocked0>
  %cst_4 = arith.constant dense<150> : tensor<1024xi32, #blocked0>
  %cst_5 = arith.constant dense<false> : tensor<1024xi1, #blocked0>
  %cst_6 = arith.constant dense<2> : tensor<1024xi32, #blocked0>
  %cst_7 = arith.constant dense<4999> : tensor<1024xi32, #blocked0>
  %cst_8 = arith.constant dense<2499> : tensor<1024xi32, #blocked0>
  %cst_9 = arith.constant dense<2500> : tensor<1024xi32, #blocked0>
  %cst_10 = arith.constant dense<0.91629076> : tensor<1024xf32, #blocked0>
  %c2499_i32 = arith.constant 2499 : i32
  %cst_11 = arith.constant dense<1024> : tensor<1024xi32, #blocked0>
  %c1024_i32 = arith.constant 1024 : i32
  %cst_12 = arith.constant dense<1> : tensor<1024xi32, #blocked0>
  %cst_13 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked0>
  %cst_14 = arith.constant dense<0> : tensor<1024xi32, #blocked0>
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c1024_i32 : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked0>
  %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked0>
  %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked0>
  %5 = arith.cmpi "slt", %4, %cst_11 : tensor<1024xi32, #blocked0>
  %6 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %8 = ttg.convert_layout %7 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0a>
  %9 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked0a>
  %10 = tt.load %8, %9 : tensor<1024x!tt.ptr<f32>, #blocked0a>
  %11 = ttg.convert_layout %10 : tensor<1024xf32, #blocked0a> -> tensor<1024xf32, #blocked0>
  %12 = tt.splat %arg7 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>, #blocked0>
  %13 = tt.addptr %12, %4 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %14 = ttg.convert_layout %13 : tensor<1024x!tt.ptr<i64>, #blocked0> -> tensor<1024x!tt.ptr<i64>, #blocked2a>
  %15 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked2a>
  %16 = tt.load %14, %15 : tensor<1024x!tt.ptr<i64>, #blocked2a>
  %17 = ttg.convert_layout %16 : tensor<1024xi64, #blocked2a> -> tensor<1024xi64, #blocked0>
  %18 = tt.splat %arg8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %19 = tt.addptr %18, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %20 = ttg.convert_layout %19 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0a>
  %21 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked0a>
  %22 = tt.load %20, %21 : tensor<1024x!tt.ptr<f32>, #blocked0a>
  %23 = ttg.convert_layout %22 : tensor<1024xf32, #blocked0a> -> tensor<1024xf32, #blocked0>
  %24 = arith.subf %cst_13, %11 : tensor<1024xf32, #blocked0>
  %25 = math.exp %24 : tensor<1024xf32, #blocked0>
  %26 = arith.sitofp %cst_12 : tensor<1024xi32, #blocked0> to tensor<1024xf32, #blocked0>
  %27 = arith.addf %25, %26 : tensor<1024xf32, #blocked0>
  %28 = arith.divf %26, %27 : tensor<1024xf32, #blocked0>
  %29 = tt.addptr %arg6, %c2499_i32 : !tt.ptr<f32>, i32
  %30 = tt.load %29 : !tt.ptr<f32>
  %31 = arith.subf %11, %cst_10 : tensor<1024xf32, #blocked0>
  %32 = arith.subf %cst_13, %31 : tensor<1024xf32, #blocked0>
  %33 = math.exp %32 : tensor<1024xf32, #blocked0>
  %34 = arith.addf %33, %26 : tensor<1024xf32, #blocked0>
  %35 = arith.divf %26, %34 : tensor<1024xf32, #blocked0>
  %36 = tt.splat %30 : f32 -> tensor<1024xf32, #blocked0>
  %37 = arith.cmpf "oge", %36, %35 : tensor<1024xf32, #blocked0>
  %38 = arith.select %37, %cst_14, %cst_9 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %39 = arith.select %37, %cst_8, %cst_7 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %40 = arith.subi %39, %38 : tensor<1024xi32, #blocked0>
  %41 = arith.cmpi "slt", %40, %cst_14 : tensor<1024xi32, #blocked0>
  %42 = arith.cmpi "ne", %41, %cst_5 : tensor<1024xi1, #blocked0>
  %43 = arith.remsi %40, %cst_6 : tensor<1024xi32, #blocked0>
  %44 = arith.cmpi "ne", %43, %cst_14 : tensor<1024xi32, #blocked0>
  %45 = arith.divsi %40, %cst_6 : tensor<1024xi32, #blocked0>
  %46 = arith.subi %45, %cst_12 : tensor<1024xi32, #blocked0>
  %47 = arith.select %44, %46, %45 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %48 = arith.select %42, %47, %45 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %49 = arith.addi %38, %48 : tensor<1024xi32, #blocked0>
  %50 = arith.cmpi "slt", %38, %39 : tensor<1024xi32, #blocked0>
  %51 = arith.select %50, %49, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %52 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %53 = tt.addptr %52, %51 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %54 = ttg.convert_layout %53 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %55 = tt.load %54 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %56 = arith.cmpf "oge", %55, %35 :tensor<1024xf32, #blocked0>
  %57 = arith.cmpi "eq", %56, %cst_5 : tensor<1024xi1, #blocked0>
  %58 = arith.andi %57, %50 : tensor<1024xi1, #blocked0>
  %59 = arith.addi %51, %cst_12 : tensor<1024xi32, #blocked0>
  %60 = arith.select %58, %59, %38 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %61 = arith.andi %56, %50 : tensor<1024xi1, #blocked0>
  %62 = arith.select %61, %51, %39 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %63 = arith.cmpi "slt", %60, %62 : tensor<1024xi32, #blocked0>
  %64 = arith.subi %62, %60 : tensor<1024xi32, #blocked0>
  %65 = arith.cmpi "slt", %64, %cst_14 : tensor<1024xi32, #blocked0>
  %66 = arith.cmpi "ne", %65, %cst_5 : tensor<1024xi1, #blocked0>
  %67 = arith.remsi %64, %cst_6 : tensor<1024xi32, #blocked0>
  %68 = arith.cmpi "ne", %67, %cst_14 : tensor<1024xi32, #blocked0>
  %69 = arith.divsi %64, %cst_6 : tensor<1024xi32, #blocked0>
  %70 = arith.subi %69, %cst_12 : tensor<1024xi32, #blocked0>
  %71 = arith.select %68, %70, %69 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %72 = arith.select %66, %71, %69 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %73 = arith.addi %60, %72 : tensor<1024xi32, #blocked0>
  %74 = arith.select %63, %73, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %75 = tt.addptr %52, %74 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %76 = ttg.convert_layout %75 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %77 = tt.load %76 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %78 = arith.cmpf "oge", %77, %35 :tensor<1024xf32, #blocked0>
  %79 = arith.cmpi "eq", %78, %cst_5 : tensor<1024xi1, #blocked0>
  %80 = arith.andi %79, %63 : tensor<1024xi1, #blocked0>
  %81 = arith.addi %74, %cst_12 : tensor<1024xi32, #blocked0>
  %82 = arith.select %80, %81, %60 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %83 = arith.andi %78, %63 : tensor<1024xi1, #blocked0>
  %84 = arith.select %83, %74, %62 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %85 = arith.cmpi "slt", %82, %84 : tensor<1024xi32, #blocked0>
  %86 = arith.subi %84, %82 : tensor<1024xi32, #blocked0>
  %87 = arith.cmpi "slt", %86, %cst_14 : tensor<1024xi32, #blocked0>
  %88 = arith.cmpi "ne", %87, %cst_5 : tensor<1024xi1, #blocked0>
  %89 = arith.remsi %86, %cst_6 : tensor<1024xi32, #blocked0>
  %90 = arith.cmpi "ne", %89, %cst_14 : tensor<1024xi32, #blocked0>
  %91 = arith.divsi %86, %cst_6 : tensor<1024xi32, #blocked0>
  %92 = arith.subi %91, %cst_12 : tensor<1024xi32, #blocked0>
  %93 = arith.select %90, %92, %91 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %94 = arith.select %88, %93, %91 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %95 = arith.addi %82, %94 : tensor<1024xi32, #blocked0>
  %96 = arith.select %85, %95, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %97 = tt.addptr %52, %96 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %98 = ttg.convert_layout %97 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %99 = tt.load %98 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %100 = arith.cmpf "oge", %99, %35 : tensor<1024xf32, #blocked0>
  %101 = arith.cmpi "eq", %100, %cst_5 : tensor<1024xi1, #blocked0>
  %102 = arith.andi %101, %85 : tensor<1024xi1, #blocked0>
  %103 = arith.addi %96, %cst_12 : tensor<1024xi32, #blocked0>
  %104 = arith.select %102, %103, %82 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %105 = arith.andi %100, %85 : tensor<1024xi1, #blocked0>
  %106 = arith.select %105, %96, %84 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %107 = arith.cmpi "slt", %104, %106 : tensor<1024xi32, #blocked0>
  %108 = arith.subi %106, %104 : tensor<1024xi32, #blocked0>
  %109 = arith.cmpi "slt", %108, %cst_14 : tensor<1024xi32, #blocked0>
  %110 = arith.cmpi "ne", %109, %cst_5 : tensor<1024xi1, #blocked0>
  %111 = arith.remsi %108, %cst_6 : tensor<1024xi32, #blocked0>
  %112 = arith.cmpi "ne", %111, %cst_14 : tensor<1024xi32, #blocked0>
  %113 = arith.divsi %108, %cst_6 : tensor<1024xi32, #blocked0>
  %114 = arith.subi %113, %cst_12 : tensor<1024xi32, #blocked0>
  %115 = arith.select %112, %114, %113 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %116 = arith.select %110, %115, %113 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %117 = arith.addi %104, %116 : tensor<1024xi32, #blocked0>
  %118 = arith.select %107, %117, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %119 = tt.addptr %52, %118 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %120 = ttg.convert_layout %119 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %121 = tt.load %120 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %122 = arith.cmpf "oge", %121, %35 : tensor<1024xf32, #blocked0>
  %123 = arith.cmpi "eq", %122, %cst_5 : tensor<1024xi1, #blocked0>
  %124 = arith.andi %123, %107 : tensor<1024xi1, #blocked0>
  %125 = arith.addi %118, %cst_12 : tensor<1024xi32, #blocked0>
  %126 = arith.select %124, %125, %104 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %127 = arith.andi %122, %107 : tensor<1024xi1, #blocked0>
  %128 = arith.select %127, %118, %106 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %129 = arith.cmpi "slt", %126, %128 : tensor<1024xi32, #blocked0>
  %130 = arith.subi %128, %126 : tensor<1024xi32, #blocked0>
  %131 = arith.cmpi "slt", %130, %cst_14 : tensor<1024xi32, #blocked0>
  %132 = arith.cmpi "ne", %131, %cst_5 : tensor<1024xi1, #blocked0>
  %133 = arith.remsi %130, %cst_6 : tensor<1024xi32, #blocked0>
  %134 = arith.cmpi "ne", %133, %cst_14 : tensor<1024xi32, #blocked0>
  %135 = arith.divsi %130, %cst_6 : tensor<1024xi32, #blocked0>
  %136 = arith.subi %135, %cst_12 : tensor<1024xi32, #blocked0>
  %137 = arith.select %134, %136, %135 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %138 = arith.select %132, %137, %135 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %139 = arith.addi %126, %138 : tensor<1024xi32, #blocked0>
  %140 = arith.select %129, %139, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %141 = tt.addptr %52, %140 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %142 = ttg.convert_layout %141 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %143 = tt.load %142 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %144 = arith.cmpf "oge", %143, %35 : tensor<1024xf32, #blocked0>
  %145 = arith.cmpi "eq", %144, %cst_5 : tensor<1024xi1, #blocked0>
  %146 = arith.andi %145, %129 : tensor<1024xi1, #blocked0>
  %147 = arith.addi %140, %cst_12 : tensor<1024xi32, #blocked0>
  %148 = arith.select %146, %147, %126 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %149 = arith.andi %144, %129 : tensor<1024xi1, #blocked0>
  %150 = arith.select %149, %140, %128 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %151 = arith.cmpi "slt", %148, %150 : tensor<1024xi32, #blocked0>
  %152 = arith.subi %150, %148 : tensor<1024xi32, #blocked0>
  %153 = arith.cmpi "slt", %152, %cst_14 : tensor<1024xi32, #blocked0>
  %154 = arith.cmpi "ne", %153, %cst_5 : tensor<1024xi1, #blocked0>
  %155 = arith.remsi %152, %cst_6 : tensor<1024xi32, #blocked0>
  %156 = arith.cmpi "ne", %155, %cst_14 : tensor<1024xi32, #blocked0>
  %157 = arith.divsi %152, %cst_6 : tensor<1024xi32, #blocked0>
  %158 = arith.subi %157, %cst_12 : tensor<1024xi32, #blocked0>
  %159 = arith.select %156, %158, %157 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %160 = arith.select %154, %159, %157 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %161 = arith.addi %148, %160 : tensor<1024xi32, #blocked0>
  %162 = arith.select %151, %161, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %163 = tt.addptr %52, %162 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %164 = ttg.convert_layout %163 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %165 = tt.load %164 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %166 = arith.cmpf "oge", %165, %35 : tensor<1024xf32, #blocked0>
  %167 = arith.cmpi "eq", %166, %cst_5 : tensor<1024xi1, #blocked0>
  %168 = arith.andi %167, %151 : tensor<1024xi1, #blocked0>
  %169 = arith.addi %162, %cst_12 : tensor<1024xi32, #blocked0>
  %170 = arith.select %168, %169, %148 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %171 = arith.andi %166, %151 : tensor<1024xi1, #blocked0>
  %172 = arith.select %171, %162, %150 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %173 = arith.cmpi "slt", %170, %172 : tensor<1024xi32, #blocked0>
  %174 = arith.subi %172, %170 : tensor<1024xi32, #blocked0>
  %175 = arith.cmpi "slt", %174, %cst_14 : tensor<1024xi32, #blocked0>
  %176 = arith.cmpi "ne", %175, %cst_5 : tensor<1024xi1, #blocked0>
  %177 = arith.remsi %174, %cst_6 : tensor<1024xi32, #blocked0>
  %178 = arith.cmpi "ne", %177, %cst_14 : tensor<1024xi32, #blocked0>
  %179 = arith.divsi %174, %cst_6 : tensor<1024xi32, #blocked0>
  %180 = arith.subi %179, %cst_12 : tensor<1024xi32, #blocked0>
  %181 = arith.select %178, %180, %179 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %182 = arith.select %176, %181, %179 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %183 = arith.addi %170, %182 : tensor<1024xi32, #blocked0>
  %184 = arith.select %173, %183, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %185 = tt.addptr %52, %184 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %186 = ttg.convert_layout %185 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %187 = tt.load %186 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %188 = arith.cmpf "oge", %187, %35 : tensor<1024xf32, #blocked0>
  %189 = arith.cmpi "eq", %188, %cst_5 : tensor<1024xi1, #blocked0>
  %190 = arith.andi %189, %173 : tensor<1024xi1, #blocked0>
  %191 = arith.addi %184, %cst_12 : tensor<1024xi32, #blocked0>
  %192 = arith.select %190, %191, %170 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %193 = arith.andi %188, %173 : tensor<1024xi1, #blocked0>
  %194 = arith.select %193, %184, %172 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %195 = arith.cmpi "slt", %192, %194 : tensor<1024xi32, #blocked0>
  %196 = arith.subi %194, %192 : tensor<1024xi32, #blocked0>
  %197 = arith.cmpi "slt", %196, %cst_14 : tensor<1024xi32, #blocked0>
  %198 = arith.cmpi "ne", %197, %cst_5 : tensor<1024xi1, #blocked0>
  %199 = arith.remsi %196, %cst_6 : tensor<1024xi32, #blocked0>
  %200 = arith.cmpi "ne", %199, %cst_14 : tensor<1024xi32, #blocked0>
  %201 = arith.divsi %196, %cst_6 : tensor<1024xi32, #blocked0>
  %202 = arith.subi %201, %cst_12 : tensor<1024xi32, #blocked0>
  %203 = arith.select %200, %202, %201 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %204 = arith.select %198, %203, %201 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %205 = arith.addi %192, %204 : tensor<1024xi32, #blocked0>
  %206 = arith.select %195, %205, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %207 = tt.addptr %52, %206 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %208 = ttg.convert_layout %207 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %209 = tt.load %208 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %210 = arith.cmpf "oge", %209, %35 :tensor<1024xf32, #blocked0>
  %211 = arith.cmpi "eq", %210, %cst_5 : tensor<1024xi1, #blocked0>
  %212 = arith.andi %211, %195 : tensor<1024xi1, #blocked0>
  %213 = arith.addi %206, %cst_12 : tensor<1024xi32, #blocked0>
  %214 = arith.select %212, %213, %192 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %215 = arith.andi %210, %195 : tensor<1024xi1, #blocked0>
  %216 = arith.select %215, %206, %194 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %217 = arith.cmpi "slt", %214, %216 : tensor<1024xi32, #blocked0>
  %218 = arith.subi %216, %214 : tensor<1024xi32, #blocked0>
  %219 = arith.cmpi "slt", %218, %cst_14 : tensor<1024xi32, #blocked0>
  %220 = arith.cmpi "ne", %219, %cst_5 : tensor<1024xi1, #blocked0>
  %221 = arith.remsi %218, %cst_6 : tensor<1024xi32, #blocked0>
  %222 = arith.cmpi "ne", %221, %cst_14 : tensor<1024xi32, #blocked0>
  %223 = arith.divsi %218, %cst_6 : tensor<1024xi32, #blocked0>
  %224 = arith.subi %223, %cst_12 : tensor<1024xi32, #blocked0>
  %225 = arith.select %222, %224, %223 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %226 = arith.select %220, %225, %223 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %227 = arith.addi %214, %226 : tensor<1024xi32, #blocked0>
  %228 = arith.select %217, %227, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %229 = tt.addptr %52, %228 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %230 = ttg.convert_layout %229 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %231 = tt.load %230 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %232 = arith.cmpf "oge", %231, %35 : tensor<1024xf32, #blocked0>
  %233 = arith.cmpi "eq", %232, %cst_5 : tensor<1024xi1, #blocked0>
  %234 = arith.andi %233, %217 : tensor<1024xi1, #blocked0>
  %235 = arith.addi %228, %cst_12 : tensor<1024xi32, #blocked0>
  %236 = arith.select %234, %235, %214 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %237 = arith.andi %232, %217 : tensor<1024xi1, #blocked0>
  %238 = arith.select %237, %228, %216 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %239 = arith.cmpi "slt", %236, %238 : tensor<1024xi32, #blocked0>
  %240 = arith.subi %238, %236 : tensor<1024xi32, #blocked0>
  %241 = arith.cmpi "slt", %240, %cst_14 : tensor<1024xi32, #blocked0>
  %242 = arith.cmpi "ne", %241, %cst_5 : tensor<1024xi1, #blocked0>
  %243 = arith.remsi %240, %cst_6 : tensor<1024xi32, #blocked0>
  %244 = arith.cmpi "ne", %243, %cst_14 : tensor<1024xi32, #blocked0>
  %245 = arith.divsi %240, %cst_6 : tensor<1024xi32, #blocked0>
  %246 = arith.subi %245, %cst_12 : tensor<1024xi32, #blocked0>
  %247 = arith.select %244, %246, %245 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %248 = arith.select %242, %247, %245 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %249 = arith.addi %236, %248 : tensor<1024xi32, #blocked0>
  %250 = arith.select %239, %249, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %251 = tt.addptr %52, %250 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %252 = ttg.convert_layout %251 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %253 = tt.load %252 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %254 = arith.cmpf "oge", %253, %35 : tensor<1024xf32, #blocked0>
  %255 = arith.cmpi "eq", %254, %cst_5 : tensor<1024xi1, #blocked0>
  %256 = arith.andi %255, %239 : tensor<1024xi1, #blocked0>
  %257 = arith.addi %250, %cst_12 : tensor<1024xi32, #blocked0>
  %258 = arith.select %256, %257, %236 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %259 = arith.andi %254, %239 : tensor<1024xi1, #blocked0>
  %260 = arith.select %259, %250, %238 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %261 = arith.cmpi "slt", %258, %260 : tensor<1024xi32, #blocked0>
  %262 = arith.subi %260, %258 : tensor<1024xi32, #blocked0>
  %263 = arith.cmpi "slt", %262, %cst_14 : tensor<1024xi32, #blocked0>
  %264 = arith.cmpi "ne", %263, %cst_5 : tensor<1024xi1, #blocked0>
  %265 = arith.remsi %262, %cst_6 : tensor<1024xi32, #blocked0>
  %266 = arith.cmpi "ne", %265, %cst_14 : tensor<1024xi32, #blocked0>
  %267 = arith.divsi %262, %cst_6 : tensor<1024xi32, #blocked0>
  %268 = arith.subi %267, %cst_12 : tensor<1024xi32, #blocked0>
  %269 = arith.select %266, %268, %267 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %270 = arith.select %264, %269, %267 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %271 = arith.addi %258, %270 : tensor<1024xi32, #blocked0>
  %272 = arith.select %261, %271, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %273 = tt.addptr %52, %272 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %274 = ttg.convert_layout %273 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %275 = tt.load %274 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %276 = arith.cmpf "oge", %275, %35 : tensor<1024xf32, #blocked0>
  %277 = arith.cmpi "eq", %276, %cst_5 : tensor<1024xi1, #blocked0>
  %278 = arith.andi %277, %261 : tensor<1024xi1, #blocked0>
  %279 = arith.addi %272, %cst_12 : tensor<1024xi32, #blocked0>
  %280 = arith.select %278, %279, %258 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %281 = arith.andi %276, %261 : tensor<1024xi1, #blocked0>
  %282 = arith.select %281, %272, %260 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %283 = arith.cmpi "slt", %280, %282 : tensor<1024xi32, #blocked0>
  %284 = arith.subi %282, %280 : tensor<1024xi32, #blocked0>
  %285 = arith.cmpi "slt", %284, %cst_14 : tensor<1024xi32, #blocked0>
  %286 = arith.cmpi "ne", %285, %cst_5 : tensor<1024xi1, #blocked0>
  %287 = arith.remsi %284, %cst_6 : tensor<1024xi32, #blocked0>
  %288 = arith.cmpi "ne", %287, %cst_14 : tensor<1024xi32, #blocked0>
  %289 = arith.divsi %284, %cst_6 : tensor<1024xi32, #blocked0>
  %290 = arith.subi %289, %cst_12 : tensor<1024xi32, #blocked0>
  %291 = arith.select %288, %290, %289 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %292 = arith.select %286, %291, %289 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %293 = arith.addi %280, %292 : tensor<1024xi32, #blocked0>
  %294 = arith.select %283, %293, %cst_14 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %295 = tt.addptr %52, %294 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %296 = ttg.convert_layout %295 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %297 = tt.load %296 : tensor<1024x!tt.ptr<f32>, #blocked0>
  %298 = arith.cmpf "oge", %297, %35 :tensor<1024xf32, #blocked0>
  %299 = arith.cmpi "eq", %298, %cst_5 : tensor<1024xi1, #blocked0>
  %300 = arith.andi %299, %283 : tensor<1024xi1, #blocked0>
  %301 = arith.addi %294, %cst_12 : tensor<1024xi32, #blocked0>
  %302 = arith.select %300, %301, %280 : tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>
  %303 = arith.extsi %cst_12 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %304 = arith.cmpi "eq", %17, %303 : tensor<1024xi64, #blocked0>
  %305 = arith.fptosi %23 : tensor<1024xf32, #blocked0> to tensor<1024xi64, #blocked0>
  %306 = arith.extsi %cst_14 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %307 = arith.cmpi "sgt", %306, %305 : tensor<1024xi64, #blocked0>
  %308 = arith.extsi %cst_4 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %309 = arith.cmpi "sgt", %305, %308 : tensor<1024xi64, #blocked0>
  %310 = arith.select %309, %306, %305 : tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>
  %311 = arith.select %307, %306, %310 : tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>
  %312 = arith.select %304, %311, %306 : tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>
  %313 = arith.extsi %cst_3 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %314 = arith.muli %312, %313 : tensor<1024xi64, #blocked0>
  %315 = arith.extsi %302 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %316 = arith.addi %315, %314 : tensor<1024xi64, #blocked0>
  %317 = arith.trunci %316 : tensor<1024xi64, #blocked0> to tensor<1024xi32, #blocked0>
  %318 = arith.extsi %317 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %319 = tt.splat %arg9 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %320 = tt.addptr %319, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %321 = ttg.convert_layout %320 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %322 = tt.load %321 : tensor<1024x!tt.ptr<f64>, #blocked0>
  %323 = arith.extf %cst_2 : tensor<1024xf32, #blocked0> to tensor<1024xf64, #blocked0>
  %324 = arith.cmpf "ogt", %322, %323 : tensor<1024xf64, #blocked0>
  %325 = tt.splat %arg10 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %326 = tt.addptr %325, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %327 = ttg.convert_layout %326 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %328 = tt.load %327 : tensor<1024x!tt.ptr<f64>, #blocked0>
  %329 = arith.divf %328, %322 : tensor<1024xf64, #blocked0>
  %330 = arith.truncf %329 : tensor<1024xf64, #blocked0> to tensor<1024xf32, #blocked0>
  %331 = arith.mulf %330, %cst_1 : tensor<1024xf32, #blocked0>
  %332 = arith.mulf %35, %cst_0 : tensor<1024xf32, #blocked0>
  %333 = arith.addf %331, %332 : tensor<1024xf32, #blocked0>
  %334 = arith.select %324, %333, %35 : tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>
  %335 = tt.addptr %319, %317 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi32, #blocked0>
  %336 = ttg.convert_layout %335 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %337 = tt.load %336 : tensor<1024x!tt.ptr<f64>, #blocked0>
  %338 = arith.extf %cst : tensor<1024xf32, #blocked0> to tensor<1024xf64, #blocked0>
  %339 = arith.mulf %337, %338 : tensor<1024xf64, #blocked0>
  %340 = tt.addptr %325, %317 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi32, #blocked0>
  %341 = ttg.convert_layout %340 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %342 = tt.load %341 : tensor<1024x!tt.ptr<f64>, #blocked0>
  %343 = arith.mulf %342, %338 : tensor<1024xf64, #blocked0>
  %344 = tt.splat %arg11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %345 = tt.addptr %344, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %346 = ttg.convert_layout %345 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0a>
  %347 = ttg.convert_layout %28 : tensor<1024xf32, #blocked0> -> tensor<1024xf32, #blocked0a>
  %348 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked0a>
  tt.store %346, %347, %348 : tensor<1024x!tt.ptr<f32>, #blocked0a>
  %349 = tt.splat %arg12 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked0>
  %350 = tt.addptr %349, %4 : tensor<1024x!tt.ptr<i32>, #blocked0>, tensor<1024xi32, #blocked0>
  %351 = ttg.convert_layout %350 : tensor<1024x!tt.ptr<i32>, #blocked0> -> tensor<1024x!tt.ptr<i32>, #blocked0a>
  %352 = ttg.convert_layout %317 : tensor<1024xi32, #blocked0> -> tensor<1024xi32, #blocked0a>
  %353 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked0a>
  tt.store %351, %352, %353 : tensor<1024x!tt.ptr<i32>, #blocked0a>
  %354 = tt.splat %arg13 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %355 = tt.addptr %354, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %356 = ttg.convert_layout %355 : tensor<1024x!tt.ptr<f32>, #blocked0> -> tensor<1024x!tt.ptr<f32>, #blocked0a>
  %357 = ttg.convert_layout %334 : tensor<1024xf32, #blocked0> -> tensor<1024xf32, #blocked0a>
  %358 = ttg.convert_layout %5 : tensor<1024xi1, #blocked0> -> tensor<1024xi1, #blocked0a>
  tt.store %356, %357, %358 : tensor<1024x!tt.ptr<f32>, #blocked0a>
  %359 = tt.splat %arg14 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %360 = tt.addptr %359, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %361 = ttg.convert_layout %360 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %362 = ttg.convert_layout %339 : tensor<1024xf64, #blocked0> -> tensor<1024xf64, #blocked0>
  tt.store %361, %362 : tensor<1024x!tt.ptr<f64>, #blocked0>
  %363 = tt.splat %arg15 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %364 = tt.addptr %363, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %365 = ttg.convert_layout %364 : tensor<1024x!tt.ptr<f64>, #blocked0> -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %366 = ttg.convert_layout %343 : tensor<1024xf64, #blocked0> -> tensor<1024xf64, #blocked0>
  tt.store %365, %366 : tensor<1024x!tt.ptr<f64>, #blocked0>
  tt.return
}
}

// A mnist model from torch inductor.
// Check if topological sort is working correct and there's no unnecessary convert
// CHECK-LABEL: mnist
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func public @mnist(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32) {
  // CHECK-NOT: ttg.convert_layout
  %cst = arith.constant dense<10> : tensor<16x1xi32, #blocked2>
  %cst_0 = arith.constant dense<10> : tensor<1x16xi32, #blocked3>
  %c16_i32 = arith.constant 16 : i32
  %cst_1 = arith.constant dense<64> : tensor<16x1xi32, #blocked2>
  %cst_2 = arith.constant dense<0xFF800000> : tensor<16x16xf32, #blocked2>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
  %cst_4 = arith.constant dense<0> : tensor<16x16xi32, #blocked2>
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked0>
  %3 = ttg.convert_layout %2 : tensor<16xi32, #blocked0> -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
  %5 = ttg.convert_layout %4 : tensor<16x1xi32, #blocked1> -> tensor<16x1xi32, #blocked2>
  %6 = tt.splat %1 : i32 -> tensor<16x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<16x1xi32, #blocked2>
  %8 = arith.cmpi "slt", %7, %cst_1 : tensor<16x1xi32, #blocked2>
  %9 = ttg.convert_layout %2 : tensor<16xi32, #blocked0> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x16xi32, #blocked3>
  %11 = arith.cmpi "slt", %10, %cst_0 : tensor<1x16xi32, #blocked3>
  %12 = arith.muli %7, %cst : tensor<16x1xi32, #blocked2>
  %13 = tt.broadcast %10 : tensor<1x16xi32, #blocked3> -> tensor<16x16xi32, #blocked3>
  %14 = ttg.convert_layout %13 : tensor<16x16xi32, #blocked3> -> tensor<16x16xi32, #blocked2>
  %15 = tt.broadcast %12 : tensor<16x1xi32, #blocked2> -> tensor<16x16xi32, #blocked2>
  %16 = arith.addi %14, %15 : tensor<16x16xi32, #blocked2>
  %17 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %19 = tt.broadcast %11 : tensor<1x16xi1, #blocked3> -> tensor<16x16xi1, #blocked3>
  %20 = ttg.convert_layout %19 : tensor<16x16xi1, #blocked3> -> tensor<16x16xi1, #blocked2>
  %21 = tt.broadcast %8 : tensor<16x1xi1, #blocked2> -> tensor<16x16xi1, #blocked2>
  %22 = arith.andi %20, %21 : tensor<16x16xi1, #blocked2>
  %23 = ttg.convert_layout %18 : tensor<16x16x!tt.ptr<f32>, #blocked2> -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %24 = ttg.convert_layout %22 : tensor<16x16xi1, #blocked2> -> tensor<16x16xi1, #blocked4>
  %25 = tt.load %23, %24 : tensor<16x16x!tt.ptr<f32>, #blocked4>
  %26 = ttg.convert_layout %25 : tensor<16x16xf32, #blocked4> -> tensor<16x16xf32, #blocked2>
  %27 = arith.cmpf "olt", %cst_2, %26 : tensor<16x16xf32, #blocked2>
  %28 = arith.andi %22, %27 : tensor<16x16xi1, #blocked2>
  %29 = arith.select %28, %26, %cst_2 : tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>
  %30 = "tt.reduce" (%29) ({
  ^bb0(%arg4: f32, %arg5: f32):
    %max = arith.maximumf %arg4, %arg5 : f32
    tt.reduce.return %max : f32
  }) {axis = 1 : i32} : (tensor<16x16xf32, #blocked2>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %31 = ttg.convert_layout %30 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16xf32, #blocked0>
  %32 = ttg.convert_layout %31 : tensor<16xf32, #blocked0> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xf32, #blocked1>
  %34 = ttg.convert_layout %33 : tensor<16x1xf32, #blocked1> -> tensor<16x1xf32, #blocked2>
  %35 = arith.sitofp %cst_4 : tensor<16x16xi32, #blocked2> to tensor<16x16xf32, #blocked2>
  %36 = arith.addf %35, %cst_3 : tensor<16x16xf32, #blocked2>
  %37 = ttg.convert_layout %18 : tensor<16x16x!tt.ptr<f32>, #blocked2> -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %38 = ttg.convert_layout %22 : tensor<16x16xi1, #blocked2> -> tensor<16x16xi1, #blocked4>
  %39 = tt.load %37, %38 : tensor<16x16x!tt.ptr<f32>, #blocked4>
  %40 = ttg.convert_layout %39 : tensor<16x16xf32, #blocked4> -> tensor<16x16xf32, #blocked2>
  %41 = tt.broadcast %34 : tensor<16x1xf32, #blocked2> -> tensor<16x16xf32, #blocked2>
  %42 = arith.subf %40, %41 : tensor<16x16xf32, #blocked2>
  %43 = math.exp %42 : tensor<16x16xf32, #blocked2>
  %44 = arith.addf %36, %43 : tensor<16x16xf32, #blocked2>
  %45 = arith.select %22, %44, %36 : tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>
  %46 = "tt.reduce" (%45) ({
  ^bb0(%arg4: f32, %arg5: f32):
    %add = arith.addf %arg4, %arg5 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<16x16xf32, #blocked2>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %47 = ttg.convert_layout %46 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16xf32, #blocked0>
  %48 = ttg.convert_layout %47 : tensor<16xf32, #blocked0> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %49 = tt.expand_dims %48 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xf32, #blocked1>
  %50 = ttg.convert_layout %49 : tensor<16x1xf32, #blocked1> -> tensor<16x1xf32, #blocked2>
  %51 = ttg.convert_layout %18 : tensor<16x16x!tt.ptr<f32>, #blocked2> -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %52 = ttg.convert_layout %22 : tensor<16x16xi1, #blocked2> -> tensor<16x16xi1, #blocked4>
  %53 = tt.load %51, %52 : tensor<16x16x!tt.ptr<f32>, #blocked4>
  %54 = ttg.convert_layout %53 : tensor<16x16xf32, #blocked4> -> tensor<16x16xf32, #blocked2>
  %55 = arith.subf %54, %41 : tensor<16x16xf32, #blocked2>
  %56 = math.log %50 : tensor<16x1xf32, #blocked2>
  %57 = tt.broadcast %56 : tensor<16x1xf32, #blocked2> -> tensor<16x16xf32, #blocked2>
  %58 = arith.subf %55, %57 : tensor<16x16xf32, #blocked2>
  %59 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %60 = tt.addptr %59, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %61 = ttg.convert_layout %60 : tensor<16x16x!tt.ptr<f32>, #blocked2> -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %62 = ttg.convert_layout %58 : tensor<16x16xf32, #blocked2> -> tensor<16x16xf32, #blocked4>
  %63 = ttg.convert_layout %22 : tensor<16x16xi1, #blocked2> -> tensor<16x16xi1, #blocked4>
  tt.store %61, %62, %63 : tensor<16x16x!tt.ptr<f32>, #blocked4>
  tt.return
}
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// cmpf and cmpi have different operands and result types
// CHECK-LABEL: cmp
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func public @cmp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
  %c64 = arith.constant 64 : i32
  %c2048 = arith.constant 2048 : i32
  %c0 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<-3.40282347E+38> : tensor<64x64xf32, #blocked2>
  %cst_0 = arith.constant dense<4194304> : tensor<64x1xi32, #blocked2>
  %cst_1 = arith.constant dense<12> : tensor<64x1xi32, #blocked2>
  %cst_2 = arith.constant dense<2048> : tensor<1x64xi32, #blocked3>
  %cst_3 = arith.constant dense<0> : tensor<64x64xi32, #blocked2>
  %cst_4 = arith.constant dense<2048> : tensor<64x1xi32, #blocked2>
  %cst_5 = arith.constant dense<49152> : tensor<64x1xi32, #blocked2>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked2>
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
  %3 = ttg.convert_layout %2 : tensor<64xi32, #blocked0> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
  %5 = ttg.convert_layout %4 : tensor<64x1xi32, #blocked1> -> tensor<64x1xi32, #blocked2>
  %6 = tt.splat %1 : i32 -> tensor<64x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<64x1xi32, #blocked2>
  %8 = arith.cmpi "slt", %7, %cst_5 : tensor<64x1xi32, #blocked2>
  %9 = ttg.convert_layout %2 : tensor<64xi32, #blocked0> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
  %11 = arith.remsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %12 = arith.divsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %13 = arith.sitofp %cst_3 : tensor<64x64xi32, #blocked2> to tensor<64x64xf32, #blocked2>
  %14 = arith.addf %13, %cst_6 : tensor<64x64xf32, #blocked2>
  %15 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %16 = tt.broadcast %15 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %17 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %18 = tt.broadcast %8 : tensor<64x1xi1, #blocked2> -> tensor<64x64xi1, #blocked2>
  %19 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %20 = tt.broadcast %19 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %21 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %22 = arith.muli %21, %cst_0 : tensor<64x1xi32, #blocked2>
  %23 = tt.broadcast %22 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %25 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %14) -> (tensor<64x64xf32, #blocked2>) : i32 {
    %45 = tt.splat %arg6 : i32 -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = arith.cmpi "slt", %46, %cst_2 : tensor<1x64xi32, #blocked3>
    %48 = tt.broadcast %46 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %49 = ttg.convert_layout %48 : tensor<64x64xi32, #blocked3> -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %16 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %17, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : tensor<1x64xi1, #blocked3> -> tensor<64x64xi1, #blocked3>
    %53 = ttg.convert_layout %52 : tensor<64x64xi1, #blocked3> -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %18 : tensor<64x64xi1, #blocked2>
    %55 = ttg.convert_layout %51 : tensor<64x64x!tt.ptr<f16>, #blocked2> -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 : tensor<64x64x!tt.ptr<f16>, #blocked4>
    %58 = ttg.convert_layout %57 : tensor<64x64xf16, #blocked4> -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %20 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %23 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %24, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = ttg.convert_layout %62 : tensor<64x64x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 : tensor<64x64x!tt.ptr<f32>, #blocked5>
    %66 = ttg.convert_layout %65 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = arith.cmpf "une", %67, %67 : tensor<64x64xf32, #blocked2>
    %69 = arith.cmpf "ogt", %67, %cst : tensor<64x64xf32, #blocked2>
    %70 = arith.select %69, %67, %cst : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %71 = arith.select %68, %67, %70 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.addf %arg7, %72 : tensor<64x64xf32, #blocked2>
    %74 = arith.select %54, %73, %arg7 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    scf.yield %74 : tensor<64x64xf32, #blocked2>
  }
  %26 = "tt.reduce" (%25) ({
  ^bb0(%arg8: f32, %arg9: f32):
    %add = arith.addf %arg8, %arg9 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<64x64xf32, #blocked2>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %27 = ttg.convert_layout %26 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64xf32, #blocked0>
  %28 = ttg.convert_layout %27 : tensor<64xf32, #blocked0> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xf32, #blocked1>
  %30 = ttg.convert_layout %29 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked2>
  %31 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %32 = tt.broadcast %31 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %33 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %34 = tt.broadcast %8 : tensor<64x1xi1, #blocked2> -> tensor<64x64xi1, #blocked2>
  %35 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %36 = tt.broadcast %35 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %37 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %38 = arith.muli %37, %cst_0 : tensor<64x1xi32, #blocked2>
  %39 = tt.broadcast %38 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %40 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %41 = tt.broadcast %30 : tensor<64x1xf32, #blocked2> -> tensor<64x64xf32, #blocked2>
  %42 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %43 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  scf.for %arg6 = %c0 to %c2048 step %c64 : i32 {
    %45 = tt.splat %arg6 : i32 -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = arith.cmpi "slt", %46, %cst_2 : tensor<1x64xi32, #blocked3>
    %48 = tt.broadcast %46 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %49 = ttg.convert_layout %48 : tensor<64x64xi32, #blocked3> -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %32 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %33, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : tensor<1x64xi1, #blocked3> -> tensor<64x64xi1, #blocked3>
    %53 = ttg.convert_layout %52 : tensor<64x64xi1, #blocked3> -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %34 : tensor<64x64xi1, #blocked2>
    %55 = ttg.convert_layout %51 : tensor<64x64x!tt.ptr<f16>, #blocked2> -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 : tensor<64x64x!tt.ptr<f16>, #blocked4>
    %58 = ttg.convert_layout %57 : tensor<64x64xf16, #blocked4> -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %36 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %39 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %40, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = ttg.convert_layout %62 : tensor<64x64x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 : tensor<64x64x!tt.ptr<f32>, #blocked5>
    %66 = ttg.convert_layout %65 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = arith.cmpf "une", %67, %67 : tensor<64x64xf32, #blocked2>
    %69 = arith.cmpf "ogt", %67, %cst : tensor<64x64xf32, #blocked2>
    %70 = arith.select %69, %67, %cst : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %71 = arith.select %68, %67, %70 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.divf %72, %41 : tensor<64x64xf32, #blocked2>
    %74 = tt.addptr %42, %50 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %75 = ttg.convert_layout %74 : tensor<64x64x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %76 = ttg.convert_layout %73 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked5>
    %77 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked5>
    tt.store %75, %76, %77 : tensor<64x64x!tt.ptr<f32>, #blocked5>
    %78 = tt.addptr %43, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %79 = arith.truncf %73 : tensor<64x64xf32, #blocked2> to tensor<64x64xf16, #blocked2>
    %80 = ttg.convert_layout %78 : tensor<64x64x!tt.ptr<f16>, #blocked2> -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %81 = ttg.convert_layout %79 : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #blocked4>
    %82 = ttg.convert_layout %54 : tensor<64x64xi1, #blocked2> -> tensor<64x64xi1, #blocked4>
    tt.store %80, %81, %82 : tensor<64x64x!tt.ptr<f16>, #blocked4>
  }
  tt.return
}
}

// -----

// Just make sure it doesn't crash on non-tensor types.
// CHECK-LABEL: if_no_tensor
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func public @if_no_tensor(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: ttg.convert_layout
  %c-1_i64 = arith.constant -1 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c-1_i32 = arith.constant -1 : i32
  %0 = tt.get_program_id x : i32
  %1 = tt.addptr %arg3, %0 : !tt.ptr<i64>, i32
  %2 = tt.load %1 : !tt.ptr<i64>
  %3 = arith.cmpi eq, %2, %c-1_i64 : i64
  %4 = arith.select %3, %c-1_i32, %arg2 : i32
  %5 = scf.if %3 -> (!tt.ptr<f32>) {
    scf.yield %arg0 : !tt.ptr<f32>
  } else {
    %10 = tt.addptr %arg0, %2 : !tt.ptr<f32>, i64
    scf.yield %10 : !tt.ptr<f32>
  }
  %6 = arith.extsi %4 : i32 to i64
  %7 = arith.cmpi slt, %2, %6 : i64
  %8 = tt.load %5, %7, %cst : !tt.ptr<f32>
  %9 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
  tt.store %9, %8 : !tt.ptr<f32>
  tt.return
}
}

// -----

// Check if the SimplifyReduceCvt rewriter pattern doesn't hang.
// CHECK-LABEL: reduce_cvt
// CHECK-NOT: ttg.convert_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt1(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<0> : tensor<1x2xi32, #blocked>
    %cst_0 = arith.constant dense<2> : tensor<1x2xi32, #blocked>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<2xi32, #blocked1> -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2xi32, #blocked>
    %3 = arith.cmpi "slt", %2, %cst_0 : tensor<1x2xi32, #blocked>
    %4 = "tt.reduce" (%cst) ({
    ^bb0(%arg3: i32, %arg4: i32):
      %add = arith.addi %arg3, %arg4 : i32
      tt.reduce.return %add : i32
    }) {axis = 1 : i32} : (tensor<1x2xi32, #blocked>) -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = ttg.convert_layout %4 : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xi32, #blocked1>
    %6 = ttg.convert_layout %5 : tensor<1xi32, #blocked1> -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<1x1xi32, #blocked2>
    %8 = ttg.convert_layout %7 : tensor<1x1xi32, #blocked2> -> tensor<1x1xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x2x!tt.ptr<i64>, #blocked>
    %10 = tt.addptr %9, %2 : tensor<1x2x!tt.ptr<i64>, #blocked>, tensor<1x2xi32, #blocked>
    %11 = tt.broadcast %8 : tensor<1x1xi32, #blocked> -> tensor<1x2xi32, #blocked>
    %12 = arith.extsi %11 : tensor<1x2xi32, #blocked> to tensor<1x2xi64, #blocked>
    %13 = ttg.convert_layout %10 : tensor<1x2x!tt.ptr<i64>, #blocked> -> tensor<1x2x!tt.ptr<i64>, #blocked3>
    %14 = ttg.convert_layout %12 : tensor<1x2xi64, #blocked> -> tensor<1x2xi64, #blocked3>
    %15 = ttg.convert_layout %3 : tensor<1x2xi1, #blocked> -> tensor<1x2xi1, #blocked3>
    tt.store %13, %14, %15 : tensor<1x2x!tt.ptr<i64>, #blocked3>
    tt.return
  }
}

// -----

// CHECK-LABEL: reduce_cvt2
// Match the reduction
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.reduce
// CHECK-SAME: axis = 1
// CHECK: (tensor<1x256xf32, #{{.*}}>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #{{.*}}}>>
// CHECK: ttg.convert_layout
// CHECK: tt.expand_dims
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x256xf32, #blocked>
    %c3136_i32 = arith.constant 3136 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<3.136000e+03> : tensor<1x1xf32, #blocked>
    %cst_1 = arith.constant dense<50176> : tensor<1x256xi32, #blocked>
    %cst_2 = arith.constant dense<196> : tensor<1x1xi32, #blocked>
    %cst_3 = arith.constant dense<196> : tensor<1x256xi32, #blocked>
    %cst_4 = arith.constant dense<3136> : tensor<1x256xi32, #blocked>
    %cst_5 = arith.constant dense<256> : tensor<1x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %2 = ttg.convert_layout %1 : tensor<1xi32, #blocked1> -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<1x1xi32, #blocked2>
    %4 = ttg.convert_layout %3 : tensor<1x1xi32, #blocked2> -> tensor<1x1xi32, #blocked>
    %5 = tt.splat %0 : i32 -> tensor<1x1xi32, #blocked>
    %6 = arith.addi %5, %4 : tensor<1x1xi32, #blocked>
    %7 = arith.cmpi "slt", %6, %cst_5 : tensor<1x1xi32, #blocked>
    %8 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %9 = ttg.convert_layout %8 : tensor<256xi32, #blocked1> -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %11 = arith.muli %6, %cst_2 : tensor<1x1xi32, #blocked>
    %12 = tt.broadcast %11 : tensor<1x1xi32, #blocked> -> tensor<1x256xi32, #blocked>
    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>, #blocked>
    %14 = tt.broadcast %7 : tensor<1x1xi1, #blocked> -> tensor<1x256xi1, #blocked>
    %15 = scf.for %arg5 = %c0_i32 to %c3136_i32 step %c256_i32 iter_args(%arg6 = %cst) -> (tensor<1x256xf32, #blocked>) : i32 {
      %43 = tt.splat %arg5 : i32 -> tensor<1x256xi32, #blocked>
      %44 = arith.addi %43, %10 : tensor<1x256xi32, #blocked>
      %45 = arith.cmpi "slt", %44, %cst_4 : tensor<1x256xi32, #blocked>
      %46 = arith.remsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %47 = arith.divsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %48 = arith.addi %46, %12 : tensor<1x256xi32, #blocked>
      %49 = arith.muli %47, %cst_1 : tensor<1x256xi32, #blocked>
      %50 = arith.addi %48, %49 : tensor<1x256xi32, #blocked>
      %51 = tt.addptr %13, %50 : tensor<1x256x!tt.ptr<f32>, #blocked>, tensor<1x256xi32, #blocked>
      %52 = arith.andi %45, %14 : tensor<1x256xi1, #blocked>
      %53 = ttg.convert_layout %51 : tensor<1x256x!tt.ptr<f32>, #blocked> -> tensor<1x256x!tt.ptr<f32>, #blocked3>
      %54 = ttg.convert_layout %52 : tensor<1x256xi1, #blocked> -> tensor<1x256xi1, #blocked3>
      %55 = ttg.convert_layout %cst : tensor<1x256xf32, #blocked> -> tensor<1x256xf32, #blocked3>
      %56 = tt.load %53, %54, %55 : tensor<1x256x!tt.ptr<f32>, #blocked3>
      %57 = ttg.convert_layout %56 : tensor<1x256xf32, #blocked3> -> tensor<1x256xf32, #blocked>
      %58 = arith.addf %arg6, %57 : tensor<1x256xf32, #blocked>
      %59 = arith.select %52, %58, %arg6 : tensor<1x256xi1, #blocked>, tensor<1x256xf32, #blocked>
      scf.yield %59 : tensor<1x256xf32, #blocked>
    }
    %16 = "tt.reduce" (%15) ({
    ^bb0(%arg7: f32, %arg8: f32):
      %add = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %add : f32

    }) {axis = 1 : i32} : (tensor<1x256xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = ttg.convert_layout %16 : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xf32, #blocked1>
    %18 = ttg.convert_layout %17 : tensor<1xf32, #blocked1> -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<1x1xf32, #blocked2>
    %20 = ttg.convert_layout %19 : tensor<1x1xf32, #blocked2> -> tensor<1x1xf32, #blocked>
    %21 = arith.divf %20, %cst_0 : tensor<1x1xf32, #blocked>
    %22 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %23 = tt.addptr %22, %6 : tensor<1x1x!tt.ptr<f32>, #blocked>, tensor<1x1xi32, #blocked>
    %24 = ttg.convert_layout %23 : tensor<1x1x!tt.ptr<f32>, #blocked> -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %25 = ttg.convert_layout %21 : tensor<1x1xf32, #blocked> -> tensor<1x1xf32, #blocked>
    %26 = ttg.convert_layout %7 : tensor<1x1xi1, #blocked> -> tensor<1x1xi1, #blocked>
    tt.store %24, %25, %26 : tensor<1x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Ensure that RematerializeForward doesn't apply when a convert has multiple uses
// CHECK-LABEL: loop_convert_multi_uses
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @loop_convert_multi_uses(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<16xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked1>
    %cst_3 = arith.constant dense<1> : tensor<16x1xi32, #blocked1>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg0 : i32
    %3 = arith.remsi %1, %arg0 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %5 = arith.muli %0, %c16_i32 : i32
    %6 = tt.splat %5 : i32 -> tensor<16xi32, #blocked>
    %7 = arith.addi %6, %4 : tensor<16xi32, #blocked>
    %8 = arith.muli %2, %arg3 : i32
    %9 = arith.muli %3, %arg4 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = ttg.convert_layout %7 : tensor<16xi32, #blocked> -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xi32, #blocked2>
    %13 = ttg.convert_layout %12 : tensor<16x1xi32, #blocked2> -> tensor<16x1xi32, #blocked1>
    %14 = tt.splat %arg6 : i32 -> tensor<16x1xi32, #blocked1>
    %15 = arith.muli %13, %14 : tensor<16x1xi32, #blocked1>
    %16 = ttg.convert_layout %4 : tensor<16xi32, #blocked> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x16xi32, #blocked3>
    %18 = tt.broadcast %15 : tensor<16x1xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %19 = tt.broadcast %17 : tensor<1x16xi32, #blocked3> -> tensor<16x16xi32, #blocked3>
    %20 = ttg.convert_layout %19 : tensor<16x16xi32, #blocked3> -> tensor<16x16xi32, #blocked1>
    %21 = arith.addi %18, %20 : tensor<16x16xi32, #blocked1>
    %22 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %23 = arith.cmpi "slt", %13, %cst_3 : tensor<16x1xi32, #blocked1>
    %24 = tt.broadcast %23 : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %25 = arith.truncf %cst_2 : tensor<16x16xf32, #blocked1> to tensor<16x16xf16, #blocked1>
    %26 = arith.muli %2, %arg11 : i32
    %27 = arith.muli %3, %arg12 : i32
    %28 = arith.addi %26, %27 : i32
    %29 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %30 = arith.cmpi "slt", %7, %cst_1 : tensor<16xi32, #blocked>
    %31 = arith.muli %2, %arg8 : i32
    %32 = arith.muli %3, %arg9 : i32
    %33 = arith.addi %31, %32 : i32
    %34 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %35:3 = scf.for %arg17 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg18 = %cst_2, %arg19 = %cst_0, %arg20 = %cst) -> (tensor<16x16xf32, #blocked1>, tensor<16xf32, #blocked>, tensor<16xf32, #blocked>)  : i32 {
      %60 = arith.muli %arg17, %arg5 : i32
      %61 = arith.addi %10, %60 : i32
      %62 = tt.splat %61 : i32 -> tensor<16x16xi32, #blocked1>
      %63 = arith.addi %62, %21 : tensor<16x16xi32, #blocked1>
      %64 = tt.addptr %22, %63 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %65 = ttg.convert_layout %64 : tensor<16x16x!tt.ptr<f16>, #blocked1> -> tensor<16x16x!tt.ptr<f16>, #blocked4>
      %66 = ttg.convert_layout %24 : tensor<16x16xi1, #blocked1> -> tensor<16x16xi1, #blocked4>
      %67 = ttg.convert_layout %25 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #blocked4>
      %68 = tt.load %65, %66, %67 : tensor<16x16x!tt.ptr<f16>, #blocked4>
      %69 = ttg.convert_layout %68 : tensor<16x16xf16, #blocked4> -> tensor<16x16xf16, #blocked1>
      %70 = arith.addi %28, %arg17 : i32
      %71 = tt.splat %70 : i32 -> tensor<16xi32, #blocked>
      %72 = arith.addi %71, %7 : tensor<16xi32, #blocked>
      %73 = tt.addptr %29, %72 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      %74 = ttg.convert_layout %73 : tensor<16x!tt.ptr<f32>, #blocked> -> tensor<16x!tt.ptr<f32>, #blocked>
      %75 = ttg.convert_layout %30 : tensor<16xi1, #blocked> -> tensor<16xi1, #blocked>
      %76 = ttg.convert_layout %cst_0 : tensor<16xf32, #blocked> -> tensor<16xf32, #blocked>
      %77 = tt.load %74, %75, %76 : tensor<16x!tt.ptr<f32>, #blocked>
      %78 = arith.addi %33, %arg17 : i32
      %79 = tt.splat %78 : i32 -> tensor<16xi32, #blocked>
      %80 = arith.addi %79, %7 : tensor<16xi32, #blocked>
      %81 = tt.addptr %34, %80 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      %82 = ttg.convert_layout %81 : tensor<16x!tt.ptr<f32>, #blocked> -> tensor<16x!tt.ptr<f32>, #blocked>
      %83 = ttg.convert_layout %30 : tensor<16xi1, #blocked> -> tensor<16xi1, #blocked>
      %84 = ttg.convert_layout %cst_0 : tensor<16xf32, #blocked> -> tensor<16xf32, #blocked>
      %85 = tt.load %82, %83, %84 : tensor<16x!tt.ptr<f32>, #blocked>
      %86 = arith.cmpf "ogt", %arg20, %85 : tensor<16xf32, #blocked>
      %87 = arith.select %86, %arg20, %85 : tensor<16xi1, #blocked>, tensor<16xf32, #blocked>
      %88 = arith.subf %arg20, %87 : tensor<16xf32, #blocked>
      %89 = math.exp %88 : tensor<16xf32, #blocked>
      %90 = arith.subf %85, %87 : tensor<16xf32, #blocked>
      %91 = math.exp %90 : tensor<16xf32, #blocked>
      %92 = arith.mulf %89, %arg19 : tensor<16xf32, #blocked>
      %93 = arith.mulf %91, %77 : tensor<16xf32, #blocked>
      %94 = arith.addf %92, %93 : tensor<16xf32, #blocked>
      %95 = arith.divf %91, %94 : tensor<16xf32, #blocked>
      %96 = arith.divf %arg19, %94 : tensor<16xf32, #blocked>
      %97 = arith.mulf %96, %89 : tensor<16xf32, #blocked>
      %98 = ttg.convert_layout %97 : tensor<16xf32, #blocked> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %99 = tt.expand_dims %98 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xf32, #blocked2>
      %100 = ttg.convert_layout %99 : tensor<16x1xf32, #blocked2> -> tensor<16x1xf32, #blocked1>
      %101 = tt.broadcast %100 : tensor<16x1xf32, #blocked1> -> tensor<16x16xf32, #blocked1>
      %102 = arith.mulf %arg18, %101 : tensor<16x16xf32, #blocked1>
      %103 = ttg.convert_layout %95 : tensor<16xf32, #blocked> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %104 = tt.expand_dims %103 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xf32, #blocked2>
      %105 = ttg.convert_layout %104 : tensor<16x1xf32, #blocked2> -> tensor<16x1xf32, #blocked1>
      %106 = tt.broadcast %105 : tensor<16x1xf32, #blocked1> -> tensor<16x16xf32, #blocked1>
      %107 = arith.extf %69 : tensor<16x16xf16, #blocked1> to tensor<16x16xf32, #blocked1>
      %108 = arith.mulf %107, %106 : tensor<16x16xf32, #blocked1>
      %109 = arith.addf %102, %108 : tensor<16x16xf32, #blocked1>
      scf.yield %109, %94, %87 : tensor<16x16xf32, #blocked1>, tensor<16xf32, #blocked>, tensor<16xf32, #blocked>
    }
    %36 = arith.muli %2, %arg14 : i32
    %37 = arith.muli %3, %arg15 : i32
    %38 = arith.addi %36, %37 : i32
    %39 = ttg.convert_layout %7 : tensor<16xi32, #blocked> -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xi32, #blocked2>
    %41 = ttg.convert_layout %40 : tensor<16x1xi32, #blocked2> -> tensor<16x1xi32, #blocked1>
    %42 = tt.splat %arg16 : i32 -> tensor<16x1xi32, #blocked1>
    %43 = arith.muli %41, %42 : tensor<16x1xi32, #blocked1>
    %44 = ttg.convert_layout %4 : tensor<16xi32, #blocked> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %45 = tt.expand_dims %44 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x16xi32, #blocked3>
    %46 = tt.broadcast %43 : tensor<16x1xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %47 = tt.broadcast %45 : tensor<1x16xi32, #blocked3> -> tensor<16x16xi32, #blocked3>
    %48 = ttg.convert_layout %47 : tensor<16x16xi32, #blocked3> -> tensor<16x16xi32, #blocked1>
    %49 = arith.addi %46, %48 : tensor<16x16xi32, #blocked1>
    %50 = tt.splat %38 : i32 -> tensor<16x16xi32, #blocked1>
    %51 = arith.addi %50, %49 : tensor<16x16xi32, #blocked1>
    %52 = tt.splat %arg13 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %53 = tt.addptr %52, %51 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %54 = arith.cmpi "slt", %41, %cst_3 : tensor<16x1xi32, #blocked1>
    %55 = tt.broadcast %54 : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %56 = arith.truncf %35#0 : tensor<16x16xf32, #blocked1> to tensor<16x16xf16, #blocked1>
    %57 = ttg.convert_layout %53 : tensor<16x16x!tt.ptr<f16>, #blocked1> -> tensor<16x16x!tt.ptr<f16>, #blocked4>
    %58 = ttg.convert_layout %56 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #blocked4>
    %59 = ttg.convert_layout %55 : tensor<16x16xi1, #blocked1> -> tensor<16x16xi1, #blocked4>
    tt.store %57, %58, %59 : tensor<16x16x!tt.ptr<f16>, #blocked4>
    tt.return
  }
}

// -----

// Check if MoveConvertOutOfLoop hangs because of adding additional conversions
// CHECK-LABEL: @loop_print
// CHECK-NOT: ttg.convert_layout
//     CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @loop_print(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<32> : tensor<32x128xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked2> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %3 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %4 = arith.muli %2, %3 : tensor<128x1xi32, #blocked1>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked2>
    %6 = ttg.convert_layout %5 : tensor<32xi32, #blocked2> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %8 = tt.broadcast %4 : tensor<128x1xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %9 = tt.broadcast %7 : tensor<1x32xi32, #blocked3> -> tensor<128x32xi32, #blocked3>
    %10 = ttg.convert_layout %9 : tensor<128x32xi32, #blocked3> -> tensor<128x32xi32, #blocked1>
    %11 = arith.addi %8, %10 : tensor<128x32xi32, #blocked1>
    %12 = ttg.convert_layout %5 : tensor<32xi32, #blocked2> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %14 = ttg.convert_layout %13 : tensor<32x1xi32, #blocked1> -> tensor<32x1xi32, #blocked>
    %15 = ttg.convert_layout %0 : tensor<128xi32, #blocked2> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %17 = tt.broadcast %14 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %18 = tt.broadcast %16 : tensor<1x128xi32, #blocked3> -> tensor<32x128xi32, #blocked3>
    %19 = ttg.convert_layout %18 : tensor<32x128xi32, #blocked3> -> tensor<32x128xi32, #blocked>
    %20 = arith.addi %17, %19 : tensor<32x128xi32, #blocked>
    %21 = arith.addi %arg5, %c31_i32 : i32
    %22 = arith.divsi %21, %c32_i32 : i32
    %23 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %24 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %25:3 = scf.for %arg7 = %c0_i32 to %22 step %c1_i32 iter_args(%arg8 = %cst_1, %arg9 = %11, %arg10 = %20) -> (f32, tensor<128x32xi32, #blocked1>, tensor<32x128xi32, #blocked>)  : i32 {
      tt.print "a_offsets: " { hex = false, isSigned = array<i32: 0> } : %arg9 : tensor<128x32xi32, #blocked1>
      %27 = tt.addptr %23, %arg9 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %28 = ttg.convert_layout %27 : tensor<128x32x!tt.ptr<f16>, #blocked1> -> tensor<128x32x!tt.ptr<f16>, #blocked4>
      %29 = tt.load %28 : tensor<128x32x!tt.ptr<f16>, #blocked4>
      %30 = ttg.convert_layout %29 : tensor<128x32xf16, #blocked4> -> tensor<128x32xf16, #blocked1>
      %31 = tt.addptr %24, %arg10 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %32 = ttg.convert_layout %31 : tensor<32x128x!tt.ptr<f16>, #blocked> -> tensor<32x128x!tt.ptr<f16>, #blocked5>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f16>, #blocked5>
      %34 = ttg.convert_layout %33 : tensor<32x128xf16, #blocked5> -> tensor<32x128xf16, #blocked>
      %35 = "tt.reduce"(%30) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<128x32xf16, #blocked1>) -> tensor<32xf16, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %36 = ttg.convert_layout %35 : tensor<32xf16, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<32xf16, #blocked2>
      %37 = "tt.reduce"(%36) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<32xf16, #blocked2>) -> f16
      %38 = "tt.reduce"(%34) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<32x128xf16, #blocked>) -> tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %39 = ttg.convert_layout %38 : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf16, #blocked2>
      %40 = "tt.reduce"(%39) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<128xf16, #blocked2>) -> f16
      %41 = arith.addf %37, %40 : f16
      %42 = arith.extf %41 : f16 to f32
      %43 = arith.addf %arg8, %42 : f32
      %44 = arith.addi %arg9, %cst_0 : tensor<128x32xi32, #blocked1>
      %45 = arith.addi %arg10, %cst : tensor<32x128xi32, #blocked>
      scf.yield %43, %44, %45 : f32, tensor<128x32xi32, #blocked1>, tensor<32x128xi32, #blocked>
    }
    %26 = arith.truncf %25#0 : f32 to f16
    tt.store %arg2, %26 : !tt.ptr<f16>
    tt.return
  }
}

// -----

// Check if SimplifyReduceCvt handles the cvt,reduce->reduce,cvt conversion but not the general push forward conversion
// CHECK-LABEL: reduce_cvt3
// CHECK: tt.dot
// CHECK-NEXT: tt.reduce
// CHECK: ttg.convert_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt3(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<32xi32, #blocked1> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %3 = ttg.convert_layout %2 : tensor<32x1xi32, #blocked2> -> tensor<32x1xi32, #blocked>
    %4 = arith.muli %3, %cst_0 : tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %7 = ttg.convert_layout %0 : tensor<32xi32, #blocked1> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %9 = tt.broadcast %6 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.broadcast %8 : tensor<1x32xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
    %11 = ttg.convert_layout %10 : tensor<32x32xi32, #blocked3> -> tensor<32x32xi32, #blocked>
    %12 = tt.addptr %9, %11 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %13 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %11 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %17 = ttg.convert_layout %12 : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked4>
    %18 = tt.load %17 : tensor<32x32x!tt.ptr<f16>, #blocked4>
    %19 = ttg.convert_layout %18 : tensor<32x32xf16, #blocked4> -> tensor<32x32xf16, #blocked>
    %20 = ttg.convert_layout %16 : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked4>
    %21 = tt.load %20 : tensor<32x32x!tt.ptr<f16>, #blocked4>
    %22 = ttg.convert_layout %21 : tensor<32x32xf16, #blocked4> -> tensor<32x32xf16, #blocked>
    %23 = ttg.local_alloc %22 : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem>
    %24 = ttg.memdesc_trans %23 {order=array<i32: 1,0>} : !ttg.memdesc<32x32xf16, #shared, #smem> -> !ttg.memdesc<32x32xf16, #shared1, #smem>
    %25 = ttg.local_load %24 : !ttg.memdesc<32x32xf16, #shared1, #smem> -> tensor<32x32xf16, #blocked>
    %26 = ttg.convert_layout %19 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>>
    %27 = ttg.convert_layout %25 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>>
    %28 = ttg.convert_layout %cst : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked5>
    %29 = tt.dot %26, %27, %28 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<32x32xf32, #blocked5>
    %30 = ttg.convert_layout %29 : tensor<32x32xf32, #blocked5> -> tensor<32x32xf32, #blocked>
    %31:2 = "tt.reduce"(%30, %11) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: i32):
      %37 = arith.cmpf "oeq", %arg3, %arg5 : f32
      %38 = arith.cmpi "slt", %arg4, %arg6 : i32
      %39 = arith.andi %37, %38 : i1
      %40 = arith.cmpf "ogt", %arg3, %arg5 : f32
      %41 = arith.ori %40, %39 : i1
      %42 = arith.select %41, %arg3, %arg5 : f32
      %43 = arith.select %41, %arg4, %arg6 : i32
      tt.reduce.return %42, %43 : f32, i32
    }) : (tensor<32x32xf32, #blocked>, tensor<32x32xi32, #blocked>) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>)
    %32 = ttg.convert_layout %31#1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xi32, #blocked1>
    %33 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked1>
    %34 = tt.addptr %33, %0 : tensor<32x!tt.ptr<i32>, #blocked1>, tensor<32xi32, #blocked1>
    %35 = ttg.convert_layout %34 : tensor<32x!tt.ptr<i32>, #blocked1> -> tensor<32x!tt.ptr<i32>, #blocked1>
    %36 = ttg.convert_layout %32 : tensor<32xi32, #blocked1> -> tensor<32xi32, #blocked1>
    tt.store %35, %36 : tensor<32x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}


// -----

// Check that we don't have extra convert for flash attention IR.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3a = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked4a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [0, 2, 1]}>
#blocked6a = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [8, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 4], order = [1, 0, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 4], order = [0, 1, 2]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_fw(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked2>
    %cst_3 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = arith.muli %1, %arg10 : i32
    %4 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %5 = arith.muli %0, %c128_i32 : i32
    %6 = arith.extsi %arg8 : i32 to i64
    %7 = arith.extsi %5 : i32 to i64
    %8 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i32
    %9 = arith.addi %arg20, %arg21 : i32
    %10 = arith.extsi %arg11 : i32 to i64
    %11 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i32
    %12 = arith.extsi %arg14 : i32 to i64
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %14 = tt.splat %5 : i32 -> tensor<128xi32, #blocked1>
    %15 = arith.addi %14, %13 : tensor<128xi32, #blocked1>
    %16 = arith.mulf %arg3, %cst_3 : f32
    %17 = tt.splat %4 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked3>
    %18 = tt.splat %7 : i64 -> tensor<128xi64, #blocked3a>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3a>
    %20 = arith.extsi %19 : tensor<128xi32, #blocked3a> to tensor<128xi64, #blocked3a>
    %21 = arith.addi %18, %20 : tensor<128xi64, #blocked3a>
    %22 = ttg.convert_layout %21 : tensor<128xi64, #blocked3a> -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4a}>>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4a}>> -> tensor<128x1xi64, #blocked4a>
    %24 = tt.splat %6 : i64 -> tensor<128x1xi64, #blocked4a>
    %25 = arith.muli %23, %24 : tensor<128x1xi64, #blocked4a>
    %26 = tt.broadcast %25 : tensor<128x1xi64, #blocked4a> -> tensor<128x64xi64, #blocked4a>
    %27 = ttg.convert_layout %26 : tensor<128x64xi64, #blocked4a> -> tensor<128x64xi64, #blocked3>
    %28 = tt.addptr %17, %27 : tensor<128x64x!tt.ptr<f16>, #blocked3>, tensor<128x64xi64, #blocked3>
    %29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3a>
    %30 = arith.extsi %29 : tensor<64xi32, #blocked3a> to tensor<64xi64, #blocked3a>
    %31 = ttg.convert_layout %30 : tensor<64xi64, #blocked3a> -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4a}>>
    %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4a}>> -> tensor<1x64xi64, #blocked4a>
    %33 = tt.broadcast %32 : tensor<1x64xi64, #blocked4a> -> tensor<128x64xi64, #blocked4a>
    %34 = ttg.convert_layout %33 : tensor<128x64xi64, #blocked4a> -> tensor<128x64xi64, #blocked3>
    %35 = tt.addptr %28, %34 : tensor<128x64x!tt.ptr<f16>, #blocked3>, tensor<128x64xi64, #blocked3>
    %36 = tt.load %35 : tensor<128x64x!tt.ptr<f16>, #blocked3>
    %37 = ttg.convert_layout %36 : tensor<128x64xf16, #blocked3> -> tensor<128x64xf16, #blocked2>
    %38 = tt.splat %16 : f32 -> tensor<128x64xf32, #blocked2>
    %39 = arith.extf %37 : tensor<128x64xf16, #blocked2> to tensor<128x64xf32, #blocked2>
    %40 = arith.mulf %39, %38 : tensor<128x64xf32, #blocked2>
    %41 = arith.truncf %40 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
// CHECK-NOT: ttg.convert_layout
//     CHECK: scf.for
// CHECK-NOT:   ttg.convert_layout
//     CHECK:   ttg.convert_layout %{{.*}} #ttg.dot_op
//     CHECK:   ttg.convert_layout %{{.*}} #ttg.dot_op
// CHECK-NOT:   ttg.convert_layout
//     CHECK:   tt.dot
// CHECK-NOT:   ttg.convert_layout
//     CHECK:   ttg.convert_layout %{{.*}} #ttg.dot_op
//     CHECK:   ttg.convert_layout %{{.*}} #ttg.dot_op
// CHECK-NOT:   ttg.convert_layout
//     CHECK:   tt.dot
//     CHECK:   scf.yield
    %42:5 = scf.for %arg22 = %c0_i32 to %9 step %c64_i32 iter_args(%arg23 = %cst_2, %arg24 = %cst_1, %arg25 = %cst_0, %arg26 = %c0_i64, %arg27 = %c0_i64) -> (tensor<128x64xf32, #blocked2>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>, i64, i64)  : i32 {
      %78 = tt.splat %8 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked6>
      %79 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked6a>
      %80 = arith.extsi %79 : tensor<64xi32, #blocked6a> to tensor<64xi64, #blocked6a>
      %81 = ttg.convert_layout %80 : tensor<64xi64, #blocked6a> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked6}>>
      %82 = tt.expand_dims %81 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked6}>> -> tensor<64x1xi64, #blocked6>
      %83 = tt.broadcast %82 : tensor<64x1xi64, #blocked6> -> tensor<64x64xi64, #blocked6>
      %84 = ttg.convert_layout %83 : tensor<64x64xi64, #blocked6> -> tensor<64x64xi64, #blocked6>
      %85 = tt.addptr %78, %84 : tensor<64x64x!tt.ptr<f16>, #blocked6>, tensor<64x64xi64, #blocked6>
      %86 = tt.splat %arg26 : i64 -> tensor<64xi64, #blocked6a>
      %87 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked6a>
      %88 = arith.extsi %87 : tensor<64xi32, #blocked6a> to tensor<64xi64, #blocked6a>
      %89 = arith.addi %86, %88 : tensor<64xi64, #blocked6a>
      %90 = ttg.convert_layout %89 : tensor<64xi64, #blocked6a> -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>>
      %91 = tt.expand_dims %90 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xi64, #blocked6>
      %92 = tt.splat %10 : i64 -> tensor<1x64xi64, #blocked6>
      %93 = arith.muli %91, %92 : tensor<1x64xi64, #blocked6>
      %94 = tt.broadcast %93 : tensor<1x64xi64, #blocked6> -> tensor<64x64xi64, #blocked6>
      %95 = ttg.convert_layout %94 : tensor<64x64xi64, #blocked6> -> tensor<64x64xi64, #blocked6>
      %96 = tt.addptr %85, %95 : tensor<64x64x!tt.ptr<f16>, #blocked6>, tensor<64x64xi64, #blocked6>
      %97 = tt.load %96 : tensor<64x64x!tt.ptr<f16>, #blocked6>
      %98 = tt.splat %11 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked3>
      %99 = tt.splat %arg27 : i64 -> tensor<64xi64, #blocked3a>
      %100 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3a>
      %101 = arith.extsi %100 : tensor<64xi32, #blocked3a> to tensor<64xi64, #blocked3a>
      %102 = arith.addi %99, %101 : tensor<64xi64, #blocked3a>
      %103 = ttg.convert_layout %102 : tensor<64xi64, #blocked3a> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %104 = tt.expand_dims %103 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi64, #blocked3>
      %105 = tt.splat %12 : i64 -> tensor<64x1xi64, #blocked3>
      %106 = arith.muli %104, %105 : tensor<64x1xi64, #blocked3>
      %107 = tt.broadcast %106 : tensor<64x1xi64, #blocked3> -> tensor<64x64xi64, #blocked3>
      %108 = ttg.convert_layout %107 : tensor<64x64xi64, #blocked3> -> tensor<64x64xi64, #blocked3>
      %109 = tt.addptr %98, %108 : tensor<64x64x!tt.ptr<f16>, #blocked3>, tensor<64x64xi64, #blocked3>
      %110 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3a>
      %111 = arith.extsi %110 : tensor<64xi32, #blocked3a> to tensor<64xi64, #blocked3a>
      %112 = ttg.convert_layout %111 : tensor<64xi64, #blocked3a> -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4a}>>
      %113 = tt.expand_dims %112 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked4a}>> -> tensor<1x64xi64, #blocked4a>
      %114 = tt.broadcast %113 : tensor<1x64xi64, #blocked4a> -> tensor<64x64xi64, #blocked4a>
      %115 = ttg.convert_layout %114 : tensor<64x64xi64, #blocked4a> -> tensor<64x64xi64, #blocked3>
      %116 = tt.addptr %109, %115 : tensor<64x64x!tt.ptr<f16>, #blocked3>, tensor<64x64xi64, #blocked3>
      %117 = tt.load %116 : tensor<64x64x!tt.ptr<f16>, #blocked3>
      %118 = ttg.convert_layout %41 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %119 = ttg.convert_layout %97 : tensor<64x64xf16, #blocked6> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %120 = tt.dot %118, %119, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf16, #blocked>
      %121 = ttg.convert_layout %120 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #blocked2>
      %122 = arith.extf %121 : tensor<128x64xf16, #blocked2> to tensor<128x64xf32, #blocked2>
      %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %153 = arith.maximumf %arg28, %arg29 : f32
        tt.reduce.return %153 : f32
      }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %124 = ttg.convert_layout %123 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128xf32, #blocked1>
      %125 = arith.maximumf %arg25, %124 : tensor<128xf32, #blocked1>
      %126 = arith.subf %arg25, %125 : tensor<128xf32, #blocked1>
      %127 = tt.extern_elementwise %126 {pure = true, libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #blocked1>
      %128 = ttg.convert_layout %125 : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>>
      %129 = tt.expand_dims %128 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1xf32, #blocked9>
      %130 = ttg.convert_layout %129 : tensor<128x1xf32, #blocked9> -> tensor<128x1xf32, #blocked2>
      %131 = tt.broadcast %130 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
      %132 = arith.subf %122, %131 : tensor<128x64xf32, #blocked2>
      %133 = tt.extern_elementwise %132 {pure = true, libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2>
      %134 = arith.mulf %arg24, %cst_1 : tensor<128xf32, #blocked1>
      %135 = arith.addf %134, %127 : tensor<128xf32, #blocked1>
      %136 = ttg.convert_layout %135 : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>>
      %137 = tt.expand_dims %136 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1xf32, #blocked9>
      %138 = ttg.convert_layout %137 : tensor<128x1xf32, #blocked9> -> tensor<128x1xf32, #blocked2>
      %139 = tt.broadcast %138 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
      %140 = arith.mulf %arg23, %139 : tensor<128x64xf32, #blocked2>
      %141 = arith.truncf %133 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %142 = ttg.convert_layout %141 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %143 = ttg.convert_layout %117 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %144 = ttg.convert_layout %140 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #blocked>
      %145 = tt.dot %142, %143, %144 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf32, #blocked>
      %146 = ttg.convert_layout %145 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked2>
      %147 = arith.mulf %arg24, %127 : tensor<128xf32, #blocked1>
      %148 = "tt.reduce"(%133) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %153 = arith.addf %arg28, %arg29 : f32
        tt.reduce.return %153 : f32
      }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %149 = ttg.convert_layout %148 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128xf32, #blocked1>
      %150 = arith.addf %147, %149 : tensor<128xf32, #blocked1>
      %151 = arith.addi %arg26, %c64_i64 : i64
      %152 = arith.addi %arg27, %c64_i64 : i64
      scf.yield %146, %150, %125, %151, %152 : tensor<128x64xf32, #blocked2>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>, i64, i64
    }
    %43 = ttg.convert_layout %42#1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1xf32, #blocked9>
    %45 = ttg.convert_layout %44 : tensor<128x1xf32, #blocked9> -> tensor<128x1xf32, #blocked2>
    %46 = tt.broadcast %45 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
    %47 = arith.divf %42#0, %46 : tensor<128x64xf32, #blocked2>
    %48 = arith.muli %1, %arg20 : i32
    %49 = tt.addptr %arg4, %48 : !tt.ptr<f32>, i32
    %50 = tt.splat %49 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %51 = tt.addptr %50, %15 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %52 = tt.extern_elementwise %42#1 {pure = true, libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_log2f"} : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #blocked1>
    %53 = arith.addf %42#2, %52 : tensor<128xf32, #blocked1>
    tt.store %51, %53 : tensor<128x!tt.ptr<f32>, #blocked1>
    %54 = tt.addptr %arg5, %2 : !tt.ptr<f16>, i32
    %55 = arith.extsi %arg17 : i32 to i64
    %56 = arith.extsi %5 : i32 to i64
    %57 = arith.truncf %47 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %58 = ttg.convert_layout %57 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #blocked3>
    %59 = tt.splat %54 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked3>
    %60 = tt.splat %56 : i64 -> tensor<128xi64, #blocked3a>
    %61 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3a>
    %62 = arith.extsi %61 : tensor<128xi32, #blocked3a> to tensor<128xi64, #blocked3a>
    %63 = arith.addi %60, %62 : tensor<128xi64, #blocked3a>
    %64 = ttg.convert_layout %63 : tensor<128xi64, #blocked3a> -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4a}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4a}>> -> tensor<128x1xi64, #blocked4a>
    %66 = tt.splat %55 : i64 -> tensor<128x1xi64, #blocked4a>
    %67 = arith.muli %65, %66 : tensor<128x1xi64, #blocked4a>
    %68 = tt.broadcast %67 : tensor<128x1xi64, #blocked4a> -> tensor<128x64xi64, #blocked4a>
    %69 = ttg.convert_layout %68 : tensor<128x64xi64, #blocked4a> -> tensor<128x64xi64, #blocked3>
    %70 = tt.addptr %59, %69 : tensor<128x64x!tt.ptr<f16>, #blocked3>, tensor<128x64xi64, #blocked3>
    %71 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3a>
    %72 = arith.extsi %71 : tensor<64xi32, #blocked3a> to tensor<64xi64, #blocked3a>
    %73 = ttg.convert_layout %72 : tensor<64xi64, #blocked3a> -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %74 = tt.expand_dims %73 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xi64, #blocked6>
    %75 = tt.broadcast %74 : tensor<1x64xi64, #blocked6> -> tensor<128x64xi64, #blocked6>
    %76 = ttg.convert_layout %75 : tensor<128x64xi64, #blocked6> -> tensor<128x64xi64, #blocked3>
    %77 = tt.addptr %70, %76 : tensor<128x64x!tt.ptr<f16>, #blocked3>, tensor<128x64xi64, #blocked3>
    tt.store %77, %58 : tensor<128x64x!tt.ptr<f16>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: axis_mismatch
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @axis_mismatch(%arg0: f32) -> tensor<1xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {
// CHECK: %[[R:.+]] = "tt.reduce"(%0) <{axis = 1 : i32}>
// CHECK: %[[C:.+]] = ttg.convert_layout %[[R]]
// CHECK: tt.return %[[C]]
  %0 = tt.splat %arg0 : f32 -> tensor<1x16xf32, #blocked>
  %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
    %60 = arith.addf %arg9, %arg10 : f32
    tt.reduce.return %60 : f32
  }) : (tensor<1x16xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %2 = ttg.convert_layout %1 : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xf32, #blocked1>
  %3 = ttg.convert_layout %2 : tensor<1xf32, #blocked1> -> tensor<1xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  tt.return %3: tensor<1xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: reduce_to_scalar
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   tt.return
tt.func @reduce_to_scalar(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>) -> (f32, i32) {
  %0 = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked>
  %1 = ttg.convert_layout %0 : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
  %3:2 = "tt.reduce"(%1, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
    %51 = arith.cmpf "oeq", %arg7, %arg9 : f32
    %52 = arith.cmpi "slt", %arg8, %arg10 : i32
    %53 = arith.andi %51, %52 : i1
    %54 = arith.cmpf "ogt", %arg7, %arg9 : f32
    %55 = arith.ori %54, %53 : i1
    %56 = arith.select %55, %arg7, %arg9 : f32
    %57 = arith.select %55, %arg8, %arg10 : i32
    tt.reduce.return %56, %57 : f32, i32
  }) : (tensor<1024xf32, #blocked1>, tensor<1024xi32, #blocked1>) -> (f32, i32)
  tt.return %3#0, %3#1: f32, i32
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: whileop
//       CHECK: %[[L:.+]] = tt.load %{{.*}} : tensor<1024x!tt.ptr<f32>, #blocked>
//       CHECK: %[[W:.+]] = scf.while (%[[I:.+]] = %[[L]], %{{.*}} = %{{.*}}) : (tensor<1024xf32, #blocked>, i1) -> tensor<1024xf32, #blocked> {
//       CHECK:   scf.condition(%{{.*}}) %[[I]] : tensor<1024xf32, #blocked>
//       CHECK: } do {
//       CHECK: ^bb0(%[[ARG1:.+]]: tensor<1024xf32, #blocked>):
//       CHECK:    %[[ADD:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : tensor<1024xf32, #blocked>
//       CHECK:    scf.yield %[[ADD]], %{{.*}} : tensor<1024xf32, #blocked>, i1
//       CHECK:  }
//       CHECK:  tt.store %{{.*}}, %[[W]] : tensor<1024x!tt.ptr<f32>, #blocked>
tt.func @whileop(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %cond: i1) {
  %0 = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked>
  %1 = ttg.convert_layout %0 : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
  %2 = scf.while (%arg0 = %1, %arg1 = %cond) : (tensor<1024xf32, #blocked1>, i1) -> (tensor<1024xf32, #blocked1>) {
      scf.condition(%arg1) %arg0 : tensor<1024xf32, #blocked1>
    } do {
    ^bb0(%arg0: tensor<1024xf32, #blocked1>):
      %4 = ttg.convert_layout %arg0 : tensor<1024xf32, #blocked1> -> tensor<1024xf32, #blocked>
      %5 = arith.addf %4, %4 : tensor<1024xf32, #blocked>
      %6 = ttg.convert_layout %5 : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
      scf.yield %6, %cond : tensor<1024xf32, #blocked1>, i1
    }
  %3 = ttg.convert_layout %2 : tensor<1024xf32, #blocked1> -> tensor<1024xf32, #blocked>
  tt.store %ptr, %3 : tensor<1024x!tt.ptr<f32>, #blocked>
  tt.return
}
}

// -----

// Suppose we have a loop which yields a value from outside the loop:
//   %x = ...
//   %y = ...
//   %z = for iter_args(%unused = %x) {
//     yield %y
//   }
//   return %z
//
// This loop returns %y if it runs 1 or more times; otherwise, it returns %x.
//
// Check that we don't transform this loop into `yield %x` on the incorrect
// theory that the yield is dead unless %x = %y.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK-LABEL @yield_outside_loop1
tt.func public @yield_outside_loop1(%arg0: i32, %arg1: i32) -> (i32) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %i = %c0 to %c5 step %c1 iter_args(%arg3 = %arg0) -> (i32) {
    scf.yield %arg1 : i32
  }

  // We should return %arg1, not %arg0.  (It would also be OK to return %0, if
  // the loop didn't get eliminated.)
  //
  // CHECK: tt.return %arg1
  tt.return %0 : i32
}  // end function

// CHECK-LABEL @yield_outside_loop2
tt.func public @yield_outside_loop2(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %i0 = arith.constant 0 : i32
  // Only yield a single value
  // CHECK: scf.yield %{{.*}} : i32
  %0, %1 = scf.for %i = %c0 to %c5 step %c1 iter_args(%arg3 = %arg0, %sum = %i0) -> (i32, i32) {
    %sum1 = arith.addi %sum, %arg3 : i32
    scf.yield %arg0, %sum1 : i32, i32
  }

  tt.return %0, %1 : i32, i32
}  // end function

}  // end module

// -----

// Check that we handle corner cases when hoisting conversions on top of extf because conversion operations on a smaller type are faster.
// For complex slices we may hoist convert on top of extf while the source of extf has multiple uses in the slice.
// In this case we want to make sure we don't replace other uses of extf source.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: [[$MMA:#.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

// CHECK-LABEL: @hoist_convert_above_extf_and_remat
  tt.func public @hoist_convert_above_extf_and_remat(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<256> : tensor<32x1xi32, #blocked1>
    %cst_1 = arith.constant dense<256> : tensor<256x1xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1.000000e-03> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %cst_3 = arith.constant dense<2.560000e+02> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #blocked3>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = tt.splat %1 : i32 -> tensor<32x1xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<32x1xi32, #blocked>
    %6 = arith.muli %5, %cst : tensor<32x1xi32, #blocked>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.expand_dims %7 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %10 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %11 = tt.broadcast %9 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %14 = arith.muli %13, %cst_1 : tensor<256x1xi32, #blocked>
    %15 = tt.broadcast %10 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %16 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %17 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %18 = scf.for %arg7 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg8 = %cst_4) -> (tensor<32x256xf32, #blocked3>)  : i32 {
      %58 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked>
      %59 = arith.addi %6, %58 : tensor<32x1xi32, #blocked>
      %60 = tt.broadcast %59 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
      %61 = arith.addi %60, %11 : tensor<32x64xi32, #blocked>
      %62 = tt.splat %arg7 : i32 -> tensor<256x1xi32, #blocked>
      %63 = arith.addi %14, %62 : tensor<256x1xi32, #blocked>
      %64 = tt.broadcast %63 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
      %65 = arith.addi %64, %15 : tensor<256x64xi32, #blocked>
      %66 = tt.addptr %16, %61 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %67 = tt.load %66 : tensor<32x64x!tt.ptr<f16>, #blocked>
      %68 = tt.addptr %17, %65 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
      %69 = tt.load %68 : tensor<256x64x!tt.ptr<f16>, #blocked>
      %70 = ttg.local_alloc %69 : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %71 = ttg.memdesc_trans %70 {order=array<i32: 1,0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
      %72 = ttg.convert_layout %67 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
      %73 = ttg.local_load %71 : !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
      %74 = ttg.convert_layout %arg8 : tensor<32x256xf32, #blocked3> -> tensor<32x256xf32, #mma>
      %75 = ttg.convert_layout %72 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %76 = ttg.convert_layout %73 : tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %77 = tt.dot %75, %76, %74 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x256xf32, #mma>
      %78 = ttg.convert_layout %77 : tensor<32x256xf32, #mma> -> tensor<32x256xf32, #blocked3>
      scf.yield %78 : tensor<32x256xf32, #blocked3>
    }
    %19 = arith.truncf %18 : tensor<32x256xf32, #blocked3> to tensor<32x256xf16, #blocked3>
    %20 = ttg.convert_layout %19 : tensor<32x256xf16, #blocked3> -> tensor<32x256xf16, #blocked2>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %23 = tt.expand_dims %21 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %24 = tt.expand_dims %22 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %25 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x256x!tt.ptr<f16>, #blocked2>
    %26 = tt.addptr %25, %23 : tensor<1x256x!tt.ptr<f16>, #blocked2>, tensor<1x256xi32, #blocked2>
    %27 = tt.load %26 : tensor<1x256x!tt.ptr<f16>, #blocked2>
    %28 = tt.broadcast %27 : tensor<1x256xf16, #blocked2> -> tensor<32x256xf16, #blocked2>
    %29 = arith.addf %20, %28 : tensor<32x256xf16, #blocked2>
// CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<1x256xf16, [[$BLOCKED]]> -> tensor<1x256xf16, [[$MMA]]>
// CHECK: %[[B:.+]] = tt.broadcast %[[A]]
// CHECK: %[[C:.+]] = arith.addf %[[B:.+]], {{.*}}
// CHECK: arith.extf %[[C]] : tensor<32x256xf16, [[$MMA]]> to tensor<32x256xf32, [[$MMA]]>
    %30 = arith.extf %29 : tensor<32x256xf16, #blocked2> to tensor<32x256xf32, #blocked2>
    %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      %58 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %58 : f32
    }) : (tensor<32x256xf32, #blocked2>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %32 = arith.divf %31, %cst_3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %33 = arith.mulf %30, %30 : tensor<32x256xf32, #blocked2>
    %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      %58 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %58 : f32
    }) : (tensor<32x256xf32, #blocked2>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %35 = arith.divf %34, %cst_3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %36 = arith.mulf %32, %32 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %37 = arith.subf %35, %36 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %38 = math.sqrt %37 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %39 = arith.addf %38, %cst_2 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %40 = tt.expand_dims %32 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xf32, #blocked2>
    %41 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xf32, #blocked2>
    %42 = tt.broadcast %40 : tensor<32x1xf32, #blocked2> -> tensor<32x256xf32, #blocked2>
    %43 = arith.subf %30, %42 : tensor<32x256xf32, #blocked2>
    %44 = tt.broadcast %41 : tensor<32x1xf32, #blocked2> -> tensor<32x256xf32, #blocked2>
    %45 = arith.divf %43, %44 : tensor<32x256xf32, #blocked2>
    %46 = arith.truncf %45 : tensor<32x256xf32, #blocked2> to tensor<32x256xf16, #blocked2>
    %47 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %49 = arith.muli %48, %cst_0 : tensor<32x1xi32, #blocked1>
    %50 = tt.splat %1 : i32 -> tensor<32x1xi32, #blocked1>
    %51 = arith.addi %50, %49 : tensor<32x1xi32, #blocked1>
    %52 = tt.broadcast %51 : tensor<32x1xi32, #blocked1> -> tensor<32x256xi32, #blocked1>
    %53 = tt.broadcast %24 : tensor<1x256xi32, #blocked1> -> tensor<32x256xi32, #blocked1>
    %54 = arith.addi %52, %53 : tensor<32x256xi32, #blocked1>
    %55 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<32x256x!tt.ptr<f16>, #blocked1>
    %56 = tt.addptr %55, %54 : tensor<32x256x!tt.ptr<f16>, #blocked1>, tensor<32x256xi32, #blocked1>
    %57 = ttg.convert_layout %46 : tensor<32x256xf16, #blocked2> -> tensor<32x256xf16, #blocked1>
    tt.store %56, %57 : tensor<32x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @backward_reduce_multiple_results
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   tt.return
  tt.func public @backward_reduce_multiple_results() -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %cst = arith.constant dense<0xFFF0000000000000> : tensor<1x32xf64, #blocked1>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %2 = ttg.convert_layout %1 : tensor<1x32xi32, #blocked2> -> tensor<1x32xi32, #blocked1>
    %3:2 = "tt.reduce"(%cst, %2) <{axis = 1 : i32}> ({
    ^bb0(%arg0: f64, %arg1: i32, %arg2: f64, %arg3: i32):
      %5 = arith.addi %arg1, %arg3 : i32
      %6 = arith.addf %arg0, %arg2 : f64
      tt.reduce.return %6, %5 : f64, i32
    }) : (tensor<1x32xf64, #blocked1>, tensor<1x32xi32, #blocked1>) -> (tensor<1xf64, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>)
    %4 = ttg.convert_layout %3#1 : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %4 : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
}
}  // end module

// -----

#blocked = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,1], threadsPerWarp = [16,2], warpsPerCTA = [1,1], order = [1,0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reshape_propagate
  tt.func public @reshape_propagate(%arg0: tensor<16x2xf32, #blocked>) -> tensor<32xf32, #blocked3> {
    // CHECK-NOT: ttg.convert_layout
    %a = ttg.convert_layout %arg0 : tensor<16x2xf32, #blocked> -> tensor<16x2xf32, #blocked1>
    %b = tt.reshape %a : tensor<16x2xf32, #blocked1> -> tensor<32xf32, #blocked2>
    %c = ttg.convert_layout %b : tensor<32xf32, #blocked2> -> tensor<32xf32, #blocked3>
    tt.return %c : tensor<32xf32, #blocked3>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,1], threadsPerWarp = [16,2], warpsPerCTA = [1,1], order = [1,0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reshape_sink_convert
  tt.func public @reshape_sink_convert(%arg0: tensor<16x2xf32, #blocked>) -> tensor<32xf32, #blocked2> {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.reshape
    // CHECK: ttg.convert_layout
    %a = ttg.convert_layout %arg0 : tensor<16x2xf32, #blocked> -> tensor<16x2xf32, #blocked1>
    %b = tt.reshape %a : tensor<16x2xf32, #blocked1> -> tensor<32xf32, #blocked2>
    tt.return %b : tensor<32xf32, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @permuting_reshape_propagate
  tt.func public @permuting_reshape_propagate(%arg0: tensor<16x2xf32, #blocked>) -> tensor<32xf16, #blocked2> {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.truncf
    // CHECK: ttg.convert_layout
    %a = tt.reshape %arg0 allow_reorder efficient_layout : tensor<16x2xf32, #blocked> -> tensor<32xf32, #blocked1>
    %b = ttg.convert_layout %a : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked2>
    %c = arith.truncf %b : tensor<32xf32, #blocked2> to tensor<32xf16, #blocked2>
    tt.return %c : tensor<32xf16, #blocked2>
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1dim1 = #ttg.slice<{dim = 1, parent = #blocked1}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: scan_propagation
tt.func @scan_propagation(%arg: tensor<1024xi32, #slice1dim1>) -> tensor<1024xi32, #slice1dim1> {
  %1 = ttg.convert_layout %arg : tensor<1024xi32, #slice1dim1> -> tensor<1024xi32, #blocked2>
  %2 = "tt.scan" (%1) ({
  ^bb0(%arg3: i32, %arg4: i32):
      %add = arith.addi %arg3, %arg4 : i32
      tt.scan.return %add : i32
  }) {axis = 1 : i32, reverse = false} : (tensor<1024xi32, #blocked2>) -> tensor<1024xi32, #blocked2>
  %3 = ttg.convert_layout %2 : tensor<1024xi32, #blocked2> -> tensor<1024xi32, #slice1dim1>
  // don't allow non blocked layout to be propagated to scan
  // CHECK: ttg.convert_layout
  // CHECK: tt.scan
  // CHECK: ttg.convert_layout
  // CHECK: tt.return
  tt.return %3: tensor<1024xi32, #slice1dim1>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: fw_propagate_for_op
  tt.func public @fw_propagate_for_op(%arg0: tensor<1024x4xi32, #blocked>, %arg1: tensor<1024x4x!tt.ptr<i32>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32

  // CHECK-NOT: ttg.convert_layout
  // CHECK: arith.muli
  // CHECK: scf.for
  // CHECK:   scf.yield
  // CHECK: ttg.convert_layout
  // CHECK: tt.store
    %0 = ttg.convert_layout %arg0 : tensor<1024x4xi32, #blocked> -> tensor<1024x4xi32, #blocked1>
    %1 = arith.muli %0, %0 : tensor<1024x4xi32, #blocked1>
    %2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %1) -> (tensor<1024x4xi32, #blocked1>)  : i32 {
      %3 = arith.addi %arg3, %arg3 : tensor<1024x4xi32, #blocked1>
      scf.yield %3 : tensor<1024x4xi32, #blocked1>
    }
    tt.store %arg1, %2 : tensor<1024x4x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @rematerialize_through_if
  tt.func public @rematerialize_through_if(%arg0: i1, %arg1: f32) -> tensor<32xf32, #blocked> {
    // CHECK: arith.constant {{.*}} : tensor<32xf32, #blocked>
    // CHECK: arith.constant {{.*}} : tensor<32xf32, #blocked>
    // CHECK: scf.if %arg0 -> (tensor<32xf32, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32xf32, #blocked1>
    %0 = tt.splat %arg1 : f32 -> tensor<32xf32, #blocked1>
    %3 = scf.if %arg0 -> (tensor<32xf32, #blocked1>) {
      %1 = arith.addf %cst, %0 : tensor<32xf32, #blocked1>
      scf.yield %1 : tensor<32xf32, #blocked1>
    } else {
      %2 = arith.addf %cst_0, %0 : tensor<32xf32, #blocked1>
      scf.yield %2 : tensor<32xf32, #blocked1>
    }
    %4 = ttg.convert_layout %3 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
    tt.return %4 : tensor<32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @rematerialize_if_inside_loop
  tt.func public @rematerialize_if_inside_loop() -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>) {
    // CHECK: arith.constant {{.*}} : tensor<32xf32, #blocked>
    // CHECK: arith.constant {{.*}} : tensor<32xf32, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[for:[0-9]*]]:2 = scf.for {{.*}} -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>)

    // CHECK-NOT: ttg.convert_layout
    // CHECK: scf.if %{{.*}} -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>)
    // CHECK-NOT: ttg.convert_layout
    // CHECK: scf.yield {{.*}} : tensor<32xf32, #blocked>, tensor<32xf32, #blocked>
    // CHECK: scf.yield {{.*}} : tensor<32xf32, #blocked>, tensor<32xf32, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return %[[for]]#1, %[[for]]#0
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %1:2 = scf.for %arg0 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg1 = %cst, %arg3 = %cst_0) -> (tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>) : i32 {
      %2 = arith.cmpi eq, %arg0, %c0_i32 : i32
      %3:2 = scf.if %2 -> (tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>) {
        scf.yield %cst, %cst_0 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>
      } else {
        %4 = arith.addf %arg1, %cst : tensor<32xf32, #blocked1>
        %5 = ttg.convert_layout %4 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
        %6 = arith.mulf %arg3, %5 : tensor<32xf32, #blocked>
        scf.yield %4, %6 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>
      }
      scf.yield %3#0, %3#1 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>
    }
    %7 = ttg.convert_layout %1#0 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
    tt.return %7, %1#1 : tensor<32xf32, #blocked>, tensor<32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: rematerialize_loop_arg
  tt.func public @rematerialize_loop_arg(%arg0: !tt.ptr<f16>) {
    // CHECK-NOT: ttg.convert_layout
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    // CHECK: scf.for %{{.*}} iter_args(%{{.*}} = %0) -> (tensor<128x64x!tt.ptr<f16>, #blocked>)
    // CHECK-NOT: ttg.convert_layout
    // CHECK: scf.yield %{{.*}} : tensor<128x64x!tt.ptr<f16>, #blocked>
    %1 = scf.for %arg1 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg2 = %0) -> (tensor<128x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %2 = tt.addptr %arg2, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %3 = ttg.convert_layout %2 : tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
      tt.store %3, %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %4 = tt.addptr %arg2, %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %5 = ttg.convert_layout %4 : tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
      tt.store %5, %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      scf.yield %2 : tensor<128x64x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: assertop
// CHECK: %[[L:.+]] = tt.load %{{.*}} : tensor<1024x!tt.ptr<i1>, #blocked>
// CHECK: tt.assert %[[L]]

tt.func @assertop(%ptr: tensor<1024x!tt.ptr<i1>, #blocked>) {
  %0 = tt.load %ptr : tensor<1024x!tt.ptr<i1>, #blocked>
  %1 = ttg.convert_layout %0 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
  tt.assert %1, "cond must be true " : tensor<1024xi1, #blocked1>
  tt.return
}
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,1], threadsPerWarp = [16,2], warpsPerCTA = [1,1], order = [1,0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_group_dot_wait_propagate
  tt.func public @warp_group_dot_wait_propagate(%arg0: tensor<16x2xf32, #blocked>) -> tensor<16x2xf32, #blocked> {
    // CHECK-NOT: ttg.convert_layout
    %a = ttg.convert_layout %arg0 : tensor<16x2xf32, #blocked> -> tensor<16x2xf32, #blocked1>
    %b = ttng.warp_group_dot_wait %a {pendings = 0 : i32} : tensor<16x2xf32, #blocked1>
    %c = ttg.convert_layout %b : tensor<16x2xf32, #blocked1> -> tensor<16x2xf32, #blocked>
    tt.return %c : tensor<16x2xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2,4], threadsPerWarp = [16,2], warpsPerCTA = [1,1], order = [1,0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4,2], threadsPerWarp = [2,16], warpsPerCTA = [1,1], order = [0,1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @trans_propagate
  tt.func public @trans_propagate(%arg0: tensor<16x2xf32, #blocked>) -> tensor<2x16xf32, #blocked2> {
    // CHECK: tt.trans
    // CHECK: ttg.convert_layout
    %a = ttg.convert_layout %arg0 : tensor<16x2xf32, #blocked> -> tensor<16x2xf32, #blocked1>
    %b = tt.trans %a {order=array<i32: 1,0>} : tensor<16x2xf32, #blocked1> -> tensor<2x16xf32, #blocked2>
    tt.return %b : tensor<2x16xf32, #blocked2>
  }
}


// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Verify that we don't hoist the convert on top of the broadcast. In general we should hoist the convert to reduce its cost
  // but because this would combine the 1st and 2nd convert and since the 1st convert is known to be a no-op this would
  // generate more expensive code.
  // CHECK-LABEL: @hoist_with_free_convert
  tt.func public @hoist_with_free_convert(%arg0: tensor<128x256xf32, #mma1>, %arg1: tensor<128x1xf32, #mma>) -> tensor<128x256xf32, #blocked> {
    // CHECK: ttg.convert_layout
    // CHECK: tt.broadcast
    // CHECK: ttg.convert_layout
    // CHECK: tt.return
    %0 = ttg.convert_layout %arg0 : tensor<128x256xf32, #mma1> -> tensor<128x256xf32, #mma>
    %1 = tt.broadcast %arg1 : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma>
    %2 = arith.addf %0, %1 : tensor<128x256xf32, #mma>
    %3 = ttg.convert_layout %2 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>
    tt.return %3 : tensor<128x256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rematerialize_loop_arg
  tt.func public @rematerialize_loop_arg() -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>, tensor<32xf32, #blocked1>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4096_i32 = arith.constant 4096 : i32
    // CHECK: %[[F:.+]]:3 = scf.for
    // CHECK:   %[[R:.+]] = arith.addf
    // CHECK:   arith.addf
    // CHECK:   scf.yield %{{.+}}, %{{.+}}, %[[R]]
    // CHECK: }
    // CHECK: tt.return %[[F]]#2, %[[F]]#1, %[[F]]#0
    %1:3 = scf.for %arg0 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg1 = %cst, %arg3 = %cst_0, %arg4 = %cst) -> (tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>, tensor<32xf32, #blocked1>) : i32 {
      %4 = arith.addf %arg1, %cst : tensor<32xf32, #blocked1>
      %5 = ttg.convert_layout %4 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
      %6 = arith.mulf %arg3, %5 : tensor<32xf32, #blocked>
      scf.yield %4, %6, %4 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>, tensor<32xf32, #blocked1>
    }
    %7 = ttg.convert_layout %1#0 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
    tt.return %7, %1#1, %1#2 : tensor<32xf32, #blocked>, tensor<32xf32, #blocked>, tensor<32xf32, #blocked1>

  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Regression test:
  // Rematerialization of multiple loop-carried variables, where one is
  // rematerialized to the same layout by multiple users.
  // Previously this didn't interact correctly with the de-duplication mechanism.
  // CHECK-LABEL: @multi_rematerialize_loop_arg
  tt.func public  @multi_rematerialize_loop_arg(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<i8>) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %1 = tt.load %0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
    %3 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    // CHECK: %[[F:.+]]:3 = scf.for {{.*}} -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>)
    // FIXME: The optimal number of conversions should be 4.
    // CHECK-COUNT-5: convert_layout
    // CHECK-NOT: convert_layout
    // CHECK:   scf.yield {{.*}} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    // CHECK: }
    // CHECK: tt.return %[[F]]#0, %[[F]]#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
     %5:3 = scf.for %arg2 = %c0_i32 to %c2048_i32 step %c64_i32 iter_args(%arg3 = %cst_2, %arg4 = %cst, %arg5 = %cst_0) -> (tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %6 = tt.load %2 : tensor<64x64x!tt.ptr<f16>, #blocked2>
      %7 = ttg.convert_layout %1 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %8 = ttg.convert_layout %6 : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.dot %7, %8, %cst_2, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %10 = tt.load %3 : tensor<128x64x!tt.ptr<i8>, #blocked>
      %11 = tt.load %4 : tensor<128x64x!tt.ptr<i8>, #blocked>
      %12 = arith.cmpi eq, %10, %11 : tensor<128x64xi8, #blocked>
      %13 = ttg.convert_layout %12 : tensor<128x64xi1, #blocked> -> tensor<128x64xi1, #mma>
      %14 = arith.select %13, %9, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
      %15 = ttg.convert_layout %14 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
      %16 = "tt.reduce"(%15) <{axis = 1 : i32}> ({
      ^bb0(%arg6: f32, %arg7: f32):
        %34 = arith.maxnumf %arg6, %arg7 : f32
        tt.reduce.return %34 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.maxnumf %arg5, %16 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = arith.cmpf oeq, %17, %cst_0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = ttg.convert_layout %18 : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xi1, #ttg.slice<{dim = 1, parent = #mma}>>
      %20 = arith.select %18, %cst, %17 : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi1, #mma>
      %22 = tt.broadcast %21 : tensor<128x1xi1, #mma> -> tensor<128x64xi1, #mma>
      %23 = arith.select %22, %cst_2, %14 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
      %24 = ttg.convert_layout %23 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
      %25 = arith.mulf %arg4, %cst : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %26 = ttg.convert_layout %25 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %27 = tt.expand_dims %26 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %28 = tt.broadcast %27 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
      %29 = arith.mulf %arg3, %28 : tensor<128x64xf32, #mma>
      %30 = ttg.convert_layout %23 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %31 = arith.mulf %arg4, %20 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %32 = "tt.reduce"(%24) <{axis = 1 : i32}> ({
      ^bb0(%arg6: f32, %arg7: f32):
        %34 = arith.addf %arg6, %arg7 : f32
        tt.reduce.return %34 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %33 = arith.addf %31, %32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %29, %33, %17 : tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    tt.return %5#1, %5#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Regression test:
  // The while loop use the result of the for loop as an argument.
  // When propagating the layout, we should only "forward" propagate the layout to the argument and the result of the while loop
  // CHECK-LABEL: @while_use_for
  tt.func public @while_use_for(%arg0: !tt.ptr<f16>, %arg3: !tt.ptr<f32>, %arg6: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i1 = arith.constant 1 : i1
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #blocked1>
    %1000 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked2>
    %1001 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %1002 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %1003 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x128x!tt.ptr<f32>, #blocked1>
    %74 = tt.load %1000 : tensor<256x64x!tt.ptr<f16>, #blocked2>
    %67:2 = scf.for %arg11 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg12 = %cst_0, %arg14 = %1001) -> (tensor<256x128xf32, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %76 = tt.load %arg14 : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %78 = ttg.convert_layout %74 : tensor<256x64xf16, #blocked2> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
      %79 = ttg.convert_layout %76 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
      %80 = ttg.convert_layout %arg12 : tensor<256x128xf32, #blocked1> -> tensor<256x128xf32, #blocked7>
      %81 = tt.dot %78, %79, %80, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<256x128xf32, #blocked7>
      %82 = ttg.convert_layout %81 : tensor<256x128xf32, #blocked7> -> tensor<256x128xf32, #blocked1>
      scf.yield %82, %arg14 : tensor<256x128xf32, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked1>
    }
    %68:2 = scf.while (%arg11 = %67#0, %arg12 = %c1_i32) : (tensor<256x128xf32, #blocked1>, i32) -> (tensor<256x128xf32, #blocked1>, i32) {
      scf.condition(%c0_i1) %arg11, %arg12 : tensor<256x128xf32, #blocked1>, i32
    } do {
    ^bb0(%arg11: tensor<256x128xf32, #blocked1>, %arg12: i32):
      %80 = ttg.convert_layout %1003 : tensor<256x128x!tt.ptr<f32>, #blocked1> -> tensor<256x128x!tt.ptr<f32>, #blocked1>
      %81 = tt.load %80 : tensor<256x128x!tt.ptr<f32>, #blocked1>
      %82 = arith.addf %arg11, %81 : tensor<256x128xf32, #blocked1>
      %83 = arith.addi %arg12, %c1_i32 : i32
      scf.yield %82, %83 : tensor<256x128xf32, #blocked1>, i32
    }
    %69 = arith.truncf %68#0 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
    %71 = ttg.convert_layout %69 : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #blocked1>
    tt.store %1002, %71 : tensor<256x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----
// Minimized reproducer for https://github.com/pytorch/pytorch/issues/130101
// Check that backward rematerialization bails out when the same tensor requires two different layouts

// CHECK-LABEL: double_remat
// CHECK: %[[res:.*]] = ttg.convert_layout
// CHECK-NEXT: tt.return %[[res]]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 2], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @double_remat() -> tensor<1x256xi32, #blocked> attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1x256xi32, #blocked1>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<1x2xi32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<1x2xi32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<1x2x1xi32, #blocked2>
    %3 = tt.broadcast %2 : tensor<1x2x1xi32, #blocked2> -> tensor<1x2x128xi32, #blocked2>
    %4 = tt.reshape %3 : tensor<1x2x128xi32, #blocked2> -> tensor<1x256xi32, #blocked1>
    %5 = tt.broadcast %2 : tensor<1x2x1xi32, #blocked2> -> tensor<2x2x64xi32, #blocked2>
    %6 = tt.reshape %5 : tensor<2x2x64xi32, #blocked2> -> tensor<1x256xi32, #blocked1>
    %7 = arith.cmpi ne, %4, %cst : tensor<1x256xi32, #blocked1>
    %8 = arith.select %7, %6, %cst : tensor<1x256xi1, #blocked1>, tensor<1x256xi32, #blocked1>
    %9 = ttg.convert_layout %8 : tensor<1x256xi32, #blocked1> -> tensor<1x256xi32, #blocked>
    tt.return %9 : tensor<1x256xi32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @if_condition_not_dead_inside_loop
  // CHECK: scf.if
  // CHECK-NOT: convert_layout
  tt.func public @if_condition_not_dead_inside_loop(%arg0: i32) -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>) {
    %true = arith.constant true
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %1:3 = scf.for %arg10 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg1 = %cst, %arg3 = %cst_0, %arg4 = %true) -> (tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>, i1) : i32 {
      %3:2 = scf.if %arg4 -> (tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>) {
        scf.yield %cst, %cst_0 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>
      } else {
        %4 = arith.addf %arg1, %cst : tensor<32xf32, #blocked1>
        %5 = ttg.convert_layout %4 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
        %6 = arith.mulf %arg3, %5 : tensor<32xf32, #blocked>
        scf.yield %4, %6 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>
      }
      %119 = arith.cmpi eq, %arg10, %arg0 : i32
      scf.yield %3#0, %3#1, %119 : tensor<32xf32, #blocked1>, tensor<32xf32, #blocked>, i1
    }
    %7 = ttg.convert_layout %1#0 : tensor<32xf32, #blocked1> -> tensor<32xf32, #blocked>
    tt.return %7, %1#1 : tensor<32xf32, #blocked>, tensor<32xf32, #blocked>
  }
}

// -----
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 32, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @dot_wait
  tt.func public @dot_wait(%arg0: tensor<64x64xf32, #mma>, %arg1: tensor<64x128xf32, #mma1>) -> (tensor<64x64xf32, #mma>, tensor<64x128xf32, #mma1>) {
    %0:2 = ttng.warp_group_dot_wait %arg0, %arg1 {pendings = 0 : i32} : tensor<64x64xf32, #mma>, tensor<64x128xf32, #mma1>
    tt.return %0#0, %0#1 : tensor<64x64xf32, #mma>, tensor<64x128xf32, #mma1>
    // CHECK: %[[W:.+]]:2 = ttng.warp_group_dot_wait
    // CHECK: tt.return %[[W]]#0, %[[W]]#1 : tensor<64x64xf32, #mma>, tensor<64x128xf32, #mma1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @split_propagation
  // CHECK-SAME: (%[[ARG:.+]]: tensor<128x64x2xf32
  //      CHECK: %[[S:.+]], %{{.+}} = tt.split %[[ARG]]
  //      CHECK: %[[C:.+]] = ttg.convert_layout %[[S]]
  //      CHECK: tt.return %[[C]]
  tt.func public @split_propagation(%arg0: tensor<128x64x2xf32, #blocked>) -> tensor<128x64xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<128x64x2xf32, #blocked> -> tensor<128x64x2xf32, #blocked2>
    %outLHS, %outRHS = tt.split %0 : tensor<128x64x2xf32, #blocked2> -> tensor<128x64xf32, #blocked1>
    tt.return %outLHS : tensor<128x64xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [2, 1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 2, 1], order = [4, 0, 1, 2, 3]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [2, 1, 16, 1, 1], warpsPerCTA = [1, 2, 2, 1, 1], order = [4, 0, 3, 2, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 0, 1, 2, 3]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: lift_convert_to_local_load
  // CHECK-NOT: convert_layout
  // CHECK: tt.return
  tt.func public @lift_convert_to_local_load(%arg0 : !ttg.memdesc<2x1x32x4x4xi8, #shared, #ttg.shared_memory, mutable>) -> tensor<2x4x32x1x4xi8, #blocked2> {
    %1 = ttg.local_load %arg0 : !ttg.memdesc<2x1x32x4x4xi8, #shared, #ttg.shared_memory, mutable> -> tensor<2x1x32x4x4xi8, #blocked>
    %2 = tt.trans %1 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked> -> tensor<2x4x32x1x4xi8, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<2x4x32x1x4xi8, #blocked1> -> tensor<2x4x32x1x4xi8, #blocked2>
    tt.return %3 : tensor<2x4x32x1x4xi8, #blocked2>
  }
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#CL = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_DOT = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: matmul_add
  tt.func @matmul_add(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %C : !tt.ptr<f32>) {
    %a_ptr_init = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
    %c_ptr_init = tt.splat %C : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #CL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #CL>
    %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %100:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #CL>) {
      %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
      %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
      %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
      %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B_DOT>
      %c = tt.dot %a, %b, %cst : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>
      %t = ttg.convert_layout %c : tensor<128x128xf32, #C> -> tensor<128x128xf32, #CL>
      // CHECK: %[[T0:.*]] = tt.dot
      // CHECK: arith.addf %{{.*}}, %[[T0]] : tensor<128x128xf32, #mma>
      %t2 = arith.addf %prev_c, %t : tensor<128x128xf32, #CL>
      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      // CHECK: scf.yield
      scf.yield %next_a_ptr, %next_b_ptr, %t2 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #CL>
    }

    // CHECK: ttg.convert_layout {{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked
    tt.store %c_ptr_init, %100#2 : tensor<128x128x!tt.ptr<f32>, #CL>
    tt.return
  }
}

// -----

// Minimized reproducer for compiler crash during remove layouts conversions pass:
// If dot result transformed into tensor with shape smaller than one MFMA instruction size, it triggers various asserts.
// This is a smoke test that checks that compiler do not crash.
//
// CHECK-LABEL: small_tensor_mfma

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @small_tensor_mfma(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
    %cst_3 = arith.constant dense<1.230000e+02> : tensor<32x16xf32, #mma1>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = "tt.reduce" (%1) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %3 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %3 : f32
    }) {axis = 1 : i32} : (tensor<32x32xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xf32, #blocked>
    %5 = tt.broadcast %4 : tensor<32x1xf32, #blocked> -> tensor<32x16xf32, #blocked>
    %6 = ttg.convert_layout %5 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
    %7 = tt.dot %cst_2, %6, %cst_3 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<32x16xf32, #mma1>
    %addr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked>
    %8 = ttg.convert_layout %7 : tensor<32x16xf32, #mma1> -> tensor<32x16xf32, #blocked>
    tt.store %addr, %8 : tensor<32x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [2, 1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 2, 1], order = [4, 0, 1, 2, 3]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [2, 1, 16, 1, 1], warpsPerCTA = [1, 2, 2, 1, 1], order = [4, 0, 3, 2, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 0, 1, 2, 3]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: lift_convert_to_local_load
  // CHECK-NOT: convert_layout
  // CHECK: tt.return
  tt.func public @lift_convert_to_local_load(%arg0 : !ttg.memdesc<2x1x32x4x4xi8, #shared, #smem, mutable>) -> tensor<2x4x32x1x4xi8, #blocked2> {
    %1 = ttg.local_load %arg0 : !ttg.memdesc<2x1x32x4x4xi8, #shared, #smem, mutable> -> tensor<2x1x32x4x4xi8, #blocked>
    %2 = tt.trans %1 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked> -> tensor<2x4x32x1x4xi8, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<2x4x32x1x4xi8, #blocked1> -> tensor<2x4x32x1x4xi8, #blocked2>
    tt.return %3 : tensor<2x4x32x1x4xi8, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

tt.func @forward_propagate_layout_gather(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked1>) -> tensor<1024x256xf32, #blocked> {
  // CHECK-LABEL: forward_propagate_layout_gather

  // CHECK-NOT: convert_layout
  %0 = ttg.convert_layout %arg0 : tensor<1024x256xi32, #blocked> -> tensor<1024x256xi32, #blocked2>
  %1 = tt.gather %arg1[%0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked1>, tensor<1024x256xi32, #blocked2>) -> tensor<1024x256xf32, #blocked2>
  %2 = ttg.convert_layout %1 : tensor<1024x256xf32, #blocked2> -> tensor<1024x256xf32, #blocked>
  tt.return %2 : tensor<1024x256xf32, #blocked>
}

tt.func @forward_only_propagation(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked1>) -> tensor<1024x256xf32, #blocked1> {
  // CHECK-LABEL: forward_only_propagation

  // CHECK-NEXT: [[GATHER:%.*]] = tt.gather
  %0 = ttg.convert_layout %arg0 : tensor<1024x256xi32, #blocked> -> tensor<1024x256xi32, #blocked2>
  %1 = tt.gather %arg1[%0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked1>, tensor<1024x256xi32, #blocked2>) -> tensor<1024x256xf32, #blocked2>
  // CHECK-NEXT: [[RES:%.*]] = ttg.convert_layout [[GATHER]] : tensor<1024x256xf32, #blocked> -> tensor<1024x256xf32, #blocked1>
  %2 = ttg.convert_layout %1 : tensor<1024x256xf32, #blocked2> -> tensor<1024x256xf32, #blocked1>
  // CHECK-NEXT: return [[RES]]
  tt.return %2 : tensor<1024x256xf32, #blocked1>
}

tt.func @backward_remat_gather_layout(%arg0: tensor<64x64xf32, #blocked1>) -> tensor<1x64xf32, #blocked1> {
  // CHECK-LABEL: backward_remat_gather_layout

  %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
  %2 = tt.gather %arg0[%1] {axis = 0 : i32} : (tensor<64x64xf32, #blocked1>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>

  // CHECK-NOT: convert_layout
  %3 = ttg.convert_layout %2 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
  tt.return %3 : tensor<1x64xf32, #blocked1>
}

tt.func @do_not_propagate(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked1>) -> tensor<1024x256xf32, #blocked> {
  // CHECK-LABEL: do_not_propagate

  %0 = ttg.convert_layout %arg0 : tensor<1024x256xi32, #blocked> -> tensor<1024x256xi32, #blocked2>
  // CHECK: tt.gather {{.*}} (tensor<128x256xf32, #blocked1>, tensor<1024x256xi32, #blocked2>) -> tensor<1024x256xf32, #blocked2>
  %1 = tt.gather %arg1[%0] {axis = 0 : i32, efficient_layout} : (tensor<128x256xf32, #blocked1>, tensor<1024x256xi32, #blocked2>) -> tensor<1024x256xf32, #blocked2>
  %2 = ttg.convert_layout %1 : tensor<1024x256xf32, #blocked2> -> tensor<1024x256xf32, #blocked>
  tt.return %2 : tensor<1024x256xf32, #blocked>
}

tt.func @do_not_remat(%arg0: tensor<64x64xf32, #blocked1>) -> tensor<1x64xf32, #blocked1> {
  // CHECK-LABEL: do_not_remat

  %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
  // CHECK: tt.gather {{.*}} (tensor<64x64xf32, #blocked1>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
  %2 = tt.gather %arg0[%1] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>

  %3 = ttg.convert_layout %2 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
  tt.return %3 : tensor<1x64xf32, #blocked1>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: reuse_layout_conversion
tt.func @reuse_layout_conversion(%arg0: tensor<64x64xf32, #blocked>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
  // CHECK-NEXT: %cst = arith.constant {{.*}} tensor<64x64xf32, #blocked>
  %cst = arith.constant dense<2.000000e+00> : tensor<64x64xf32, #blocked1>
  // CHECK-NEXT: [[TRANS:%.*]] = tt.trans %arg0 {{.*}} tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  %0 = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  // CHECK-NEXT: [[CVT:%.*]] = ttg.convert_layout [[TRANS]] : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
  %1 = ttg.convert_layout %0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
  // CHECK-NEXT: [[RESULT:%.*]] = arith.mulf [[CVT]], %cst : tensor<64x64xf32, #blocked>
  %2 = arith.mulf %0, %cst : tensor<64x64xf32, #blocked1>
  %3 = ttg.convert_layout %2 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
  // CHECK-NEXT: return [[CVT]], [[RESULT]]
  tt.return %1, %3 : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
}

// CHECK-LABEL: respect_dominance
tt.func @respect_dominance(%arg0: tensor<64x64xf32, #blocked>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
  %cst = arith.constant dense<2.000000e+00> : tensor<64x64xf32, #blocked1>

  // CHECK-COUNT-2: convert_layout
  %0 = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>

  %2 = arith.mulf %0, %cst : tensor<64x64xf32, #blocked1>
  %1 = ttg.convert_layout %0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
  %3 = ttg.convert_layout %2 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
  tt.return %1, %3 : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
}

// CHECK-LABEL: remat_across_regions
tt.func @remat_across_regions(%arg0: i1, %arg1: tensor<8x8xf32, #blocked>) {
  // CHECK-NEXT: scf.if
  scf.if %arg0 {
    // CHECK-NEXT: convert_layout
    %0 = ttg.convert_layout %arg1 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
    "test.keep"(%0) : (tensor<8x8xf32, #blocked1>) -> ()
  // CHECK: else
  } else {
    %0 = "test.dummy"() : () -> i32
    // CHECK: convert_layout
    %1 = ttg.convert_layout %arg1 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
    "test.keep"(%1) : (tensor<8x8xf32, #blocked1>) -> ()
  // CHECK: }
  }
  // CHECK-NEXT: return
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @hoist_one_conditional
tt.func @hoist_one_conditional(
    %arg0: i1,
    %arg1: tensor<128x32x!tt.ptr<f32>, #blocked>
) -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> {

  // CHECK: arith.constant {{.*}} tensor<128x32xf32, #blocked>
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked>
  // CHECK: scf.if
  %0 = scf.if %arg0 -> (tensor<128x32xf32, #blocked>) {
    // CHECK-NEXT: [[RES:%.*]] = tt.load
    %3 = tt.load %arg1 : tensor<128x32x!tt.ptr<f32>, #blocked>
    // CHECK-NEXT: yield [[RES]]
    scf.yield %3 : tensor<128x32xf32, #blocked>
  } else {
    scf.yield %cst : tensor<128x32xf32, #blocked>
  }
  // CHECK: [[TRUNC:%.*]] = arith.truncf
  %1 = arith.truncf %0 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
  // CHECK-NEXT: convert_layout [[TRUNC]]
  %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  tt.return %2 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
}

// CHECK-LABEL: @hoist_multiple_conditional
tt.func @hoist_multiple_conditional(
    %arg0: i1,
    %arg1: i1,
    %arg2: tensor<128x32x!tt.ptr<f32>, #blocked>,
    %arg3: tensor<128x32x!tt.ptr<f32>, #blocked>,
    %arg4: tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>,
    %arg5: tensor<128x128xf32, #mma>
) -> tensor<128x128xf32, #mma> {
  // CHECK-COUNT-1: ttg.convert_layout
  %cst0 = arith.constant dense<1.0> : tensor<128x32xf32, #blocked>
  %cst1 = arith.constant dense<2.0> : tensor<128x32xf32, #blocked>
  %0 = scf.if %arg0 -> (tensor<128x32xf32, #blocked>) {
    %3 = tt.load %arg2 : tensor<128x32x!tt.ptr<f32>, #blocked>
    scf.yield %3 : tensor<128x32xf32, #blocked>
  } else {
    scf.yield %cst0 : tensor<128x32xf32, #blocked>
  }
  %1 = scf.if %arg1 -> (tensor<128x32xf32, #blocked>) {
    %4 = tt.load %arg3 : tensor<128x32x!tt.ptr<f32>, #blocked>
    scf.yield %4 : tensor<128x32xf32, #blocked>
  } else {
    scf.yield %cst1 : tensor<128x32xf32, #blocked>
  }
  %2 = arith.addf %0, %1 : tensor<128x32xf32, #blocked>
  %3 = ttg.convert_layout %2 : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %4 = tt.dot %3, %arg4, %arg5 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
  tt.return %4 : tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @hoist_across_loop
tt.func @hoist_across_loop(
    %arg0: i1,
    %arg1: tensor<128x32x!tt.ptr<f32>, #blocked>,
    %arg2: tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>,
    %arg3: tensor<128x128xf32, #mma>
) -> tensor<128x128xf32, #mma> {
  // CHECK: arith.constant {{.*}} tensor<128x32xf32, #ttg.dot_op
  %cst = arith.constant dense<1.0> : tensor<128x32xf32, #blocked>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  // CHECK: scf.for
  %0:2 = scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg4 = %cst, %acc = %arg3) -> (tensor<128x32xf32, #blocked>, tensor<128x128xf32, #mma>) : i32 {
    // CHECK-NEXT: scf.if
    %1 = scf.if %arg0 -> (tensor<128x32xf32, #blocked>) {
      // CHECK-NEXT: [[RES:%.*]] = tt.load
      // CHECK-NEXT: ttg.convert_layout [[RES]]
      %3 = tt.load %arg1 : tensor<128x32x!tt.ptr<f32>, #blocked>
      scf.yield %3 : tensor<128x32xf32, #blocked>
    } else {
      scf.yield %arg4 : tensor<128x32xf32, #blocked>
    }
    // CHECK-NOT: ttg.convert_layout
    %2 = ttg.convert_layout %1 : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %3 = tt.dot %2, %arg2, %acc : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    scf.yield %1, %3 : tensor<128x32xf32, #blocked>, tensor<128x128xf32, #mma>
  }
  tt.return %0#1 : tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @chained_if
tt.func @chained_if(%arg0: i1, %arg1: i1, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg3: tensor<32x32x!tt.ptr<f32>, #blocked>) -> tensor<32x32xf32, #mma> {
  // CHECK-COUNT-1: ttg.convert_layout
  %cst = arith.constant dense<1.0> : tensor<32x32xf32, #blocked>
  %0 = scf.if %arg0 -> tensor<32x32xf32, #blocked> {
    %anchor = tt.load %arg2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    scf.yield %anchor : tensor<32x32xf32, #blocked>
  } else {
    scf.yield %cst : tensor<32x32xf32, #blocked>
  }
  %1 = scf.if %arg1 -> tensor<32x32xf32, #blocked> {
    %anchor = tt.load %arg3 : tensor<32x32x!tt.ptr<f32>, #blocked>
    scf.yield %anchor : tensor<32x32xf32, #blocked>
  } else {
    scf.yield %0 : tensor<32x32xf32, #blocked>
  }
  %2 = ttg.convert_layout %1 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
  tt.return %2 : tensor<32x32xf32, #mma>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @cvt_in_peeled_prologue
tt.func @cvt_in_peeled_prologue(%arg0: tensor<32x32x!tt.ptr<bf16>, #blocked>, %arg1: i1, %arg2: i32, %arg3: i32, %arg4: i1) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<0.0> : tensor<32x32xbf16, #blocked1>

  // CHECK: scf.if
  %0 = scf.if %arg1 -> (tensor<32x32xbf16, #blocked1>) {
    // CHECK-NEXT: tt.load
    %1 = tt.load %arg0 : tensor<32x32x!tt.ptr<bf16>, #blocked>
    %2 = ttg.convert_layout %1 : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #blocked1>
    // CHECK-NEXT: yield
    scf.yield %2 : tensor<32x32xbf16, #blocked1>
    // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: yield
    scf.yield %cst : tensor<32x32xbf16, #blocked1>
  // CHECK-NEXT: }
  }

  // CHECK: [[PEEL1:%.*]] = scf.if
  %1 = scf.if %arg4 -> (tensor<32x32xbf16, #blocked1>) {
    // CHECK-NEXT: tt.load
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<bf16>, #blocked>
    %3 = ttg.convert_layout %2 : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #blocked1>
    // CHECK-NEXT: yield
    scf.yield %3 : tensor<32x32xbf16, #blocked1>
    // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: yield
    scf.yield %0 : tensor<32x32xbf16, #blocked1>
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: [[CVT:%.*]] = ttg.convert_layout [[PEEL1]]
  // CHECK-NEXT: scf.for {{.*}} iter_args(%{{arg[0-9]+}} = [[CVT]])
  %3 = scf.for %i = %arg2 to %arg3 step %c1_i32 iter_args(%k = %1) -> (tensor<32x32xbf16, #blocked1>) : i32 {
    // CHECK-NEXT: scf.if
    %4 = scf.if %arg1 -> (tensor<32x32xbf16, #blocked1>) {
      // CHECK-NEXT: tt.load
      %5 = tt.load %arg0 : tensor<32x32x!tt.ptr<bf16>, #blocked>
      // CHECK-NEXT: ttg.convert_layout
      %6 = ttg.convert_layout %5 : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #blocked1>
      scf.yield %6 : tensor<32x32xbf16, #blocked1>
    } else {
      scf.yield %k : tensor<32x32xbf16, #blocked1>
    }
    "use.it"(%4) : (tensor<32x32xbf16, #blocked1>) -> ()
    scf.yield %4 : tensor<32x32xbf16, #blocked1>
  }
  // CHECK-NOT: ttg.convert_layout
  tt.return
}

// CHECK-LABEL: @cvt_in_loop_if_slice
tt.func @cvt_in_loop_if_slice(%arg0: tensor<32x32x!tt.ptr<bf16>, #blocked>, %arg1: i1, %arg2: i32, %arg3: i32, %arg4: i1) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<0.0> : tensor<32x32xbf16, #blocked>

  // CHECK: [[IF_OUT:%.*]] = scf.if
  %0 = scf.if %arg1 -> (tensor<32x32xbf16, #blocked>) {
    // CHECK-NEXT: tt.load
    %1 = tt.load %arg0 : tensor<32x32x!tt.ptr<bf16>, #blocked>
    // CHECK-NEXT: yield
    scf.yield %1 : tensor<32x32xbf16, #blocked>
    // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: yield
    scf.yield %cst : tensor<32x32xbf16, #blocked>
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: [[CVT:%.*]] = ttg.convert_layout [[IF_OUT]]
  // CHECK-NEXT: scf.for
  %1 = scf.for %i = %arg2 to %arg3 step %c1_i32 iter_args(%k = %cst) -> tensor<32x32xbf16, #blocked> : i32 {
    // CHECK-NEXT: scf.if
    %4 = scf.if %arg4 -> (tensor<32x32xbf16, #blocked>) {
      // CHECK-NEXT: tt.load
      %5 = tt.load %arg0 : tensor<32x32x!tt.ptr<bf16>, #blocked>
      // CHECK-NEXT: ttg.convert_layout
      scf.yield %5 : tensor<32x32xbf16, #blocked>
    } else {
      scf.yield %k : tensor<32x32xbf16, #blocked>
    }
    %6 = arith.addf %4, %0 : tensor<32x32xbf16, #blocked>
    // CHECK-NOT: ttg.convert_layout
    %7 = ttg.convert_layout %6 : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #blocked1>
    "use.it"(%7) : (tensor<32x32xbf16, #blocked1>) -> ()
    scf.yield %6 : tensor<32x32xbf16, #blocked>
  }

  tt.return
}

}

// -----

#linear = #ttg.linear<{register = [[1, 0], [0, 8], [0, 16]], lane = [[2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 2], [0, 4]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32}  {

// CHECK-LABEL: reduce_linear_layouts
tt.func @reduce_linear_layouts(%arg0: tensor<32x32xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> {
  // CHECK-NOT: convert_layout
  %0 = ttg.convert_layout %arg0 : tensor<32x32xi32, #linear> -> tensor<32x32xi32, #blocked>
  // CHECK-NEXT: tt.reduce
  %1 = "tt.reduce" (%0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    tt.reduce.return %arg1 : i32
  // CHECK: (tensor<32x32xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>
  }) {axis = 1 : i32} : (tensor<32x32xi32, #blocked>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %2 = ttg.convert_layout %1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
  tt.return %2 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[16, 0]], lane = [[0, 1], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // Test that after dot_scaled with rhs scales is decomposed, we are able to get rid of the redundant convert_layout
  // CHECK-LABEL: dot_scale_transpose
  tt.func public @dot_scale_transpose(%arg0: tensor<128x64xf8E4M3FN, #blocked>, %arg1: tensor<32x32xi8, #blocked1>, %arg2: tensor<128x32x!tt.ptr<bf16>, #blocked3>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<128x32xf32, #blocked1>)  : i32 {
      %3 = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<128x64xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked4>
      %4 = tt.trans %arg1 {order = array<i32: 1, 0>} : tensor<32x32xi8, #blocked1> -> tensor<32x32xi8, #blocked5>
      %5 = tt.trans %arg5 {order = array<i32: 1, 0>} : tensor<128x32xf32, #blocked1> -> tensor<32x128xf32, #blocked5>
      %6 = ttg.convert_layout %5 : tensor<32x128xf32, #blocked5> -> tensor<32x128xf32, #mma>
      %7 = ttg.convert_layout %4 : tensor<32x32xi8, #blocked5> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %9 = ttg.fp4_to_fp %7 {axis = 1 : i32} : tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = ttg.convert_layout %3 : tensor<64x128xf8E4M3FN, #blocked4> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %11 = tt.fp_to_fp %10 : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %12 = tt.dot %9, %11, %6 : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x128xf32, #mma>
      // CHECK: tt.dot
      // CHECK-NOT: ttg.convert_layout
      // CHECK: scf.yield
      %13 = ttg.convert_layout %12 : tensor<32x128xf32, #mma> -> tensor<32x128xf32, #blocked5>
      %14 = tt.trans %13 {order = array<i32: 1, 0>} : tensor<32x128xf32, #blocked5> -> tensor<128x32xf32, #blocked1>
      scf.yield %14 : tensor<128x32xf32, #blocked1>
    }
    // CHECK: arith.truncf
    // CHECK-NEXT: ttg.convert_layout
    // CHECK-NEXT: tt.store
    %1 = arith.truncf %0 : tensor<128x32xf32, #blocked1> to tensor<128x32xbf16, #blocked1>
    %2 = ttg.convert_layout %1 : tensor<128x32xbf16, #blocked1> -> tensor<128x32xbf16, #blocked3>
    tt.store %arg2, %2 : tensor<128x32x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

tt.func public @reshape_slice_dot_enc(%arg0: tensor<4x16xi32, #blocked>) -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> {
  %0 = tt.reshape %arg0 : tensor<4x16xi32, #blocked> -> tensor<64xi32, #blocked2>
  %1 = ttg.convert_layout %0 : tensor<64xi32, #blocked2> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
  %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
  %3 = ttg.convert_layout %2 : tensor<64x1xi32, #blocked3> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
  tt.return %3 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
}

}
#Cv2 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#Av2k1 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=1}>
#Bv2k1 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=1}>
#Av2k2 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=2}>
#Bv2k2 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=2}>
#Av2k4 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=4}>
#Bv2k4 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=4}>
#ALR = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALC = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#BLR = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLC = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: tt.func @push_elementwise
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[BCVT:.*]] = ttg.convert_layout %{{.*}} : {{.*}} tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: %[[C:.*]] = tt.dot %[[AF16]], %[[BCVT]]
// CHECK-SAME: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = ttg.convert_layout %a : tensor<16x16xf16, #ALR> -> tensor<16x16xf16, #Av2k4>
  %dotb = ttg.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// CHECK: tt.func @succeeds_if_arg_is_not_convert_layout
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @succeeds_if_arg_is_not_convert_layout(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %dotai8 = ttg.convert_layout %ai8 : tensor<16x16xi8, #ALR> -> tensor<16x16xi8, #Av2k4>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #BLC>
  %dotaf8 = tt.bitcast %dotai8 : tensor<16x16xi8, #Av2k4> -> tensor<16x16xf8E5M2, #Av2k4>
  %dota = tt.fp_to_fp %dotaf8 : tensor<16x16xf8E5M2, #Av2k4> -> tensor<16x16xf16, #Av2k4>
  %dotb = ttg.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}

// CHECK: tt.func @push_inline_asm_op
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.elementwise_inline_asm {{.*}} %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_inline_asm_op(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %dotb: tensor<16x16xf16, #Bv2k4>,
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %dotaf8 = tt.bitcast %ai8 : tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %dota = tt.elementwise_inline_asm "{ cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; }" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %dotaf8 : tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota_cvt = ttg.convert_layout %dota : tensor<16x16xf16, #ALR> -> tensor<16x16xf16, #Av2k4>
  %newc = tt.dot %dota_cvt, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}
}

// -----

#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: #[[BA:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @push_convert_both_operands
// CHECK-DAG: %[[ALOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BA]]>
// CHECK-DAG: %[[BLOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BB]]>
// CHECK-DAG: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[BCVT:.*]] = ttg.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: tt.dot %[[AEXT]], %[[BEXT]], %{{.*}} : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @push_convert_both_operands(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blockedA>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %al = ttg.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = ttg.convert_layout %be : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}

}

// -----

#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: #[[BA:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @update_kwidth_slice
// CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[ALOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BA]]>
// CHECK-DAG: %[[BLOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BB]]>
// CHECK-DAG: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[BCVT:.*]] = ttg.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: %[[ADD:.+]] = arith.addf %[[BEXT]], %[[CST]] : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK-DAG: tt.dot %[[AEXT]], %[[ADD]], %{{.*}} : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @update_kwidth_slice(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %cst = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blockedB>
  %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blockedA>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %add = arith.addf %be, %cst : tensor<16x16xf32, #blockedB>
  %al = ttg.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = ttg.convert_layout %add : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: tt.func @propagate_dot_op_to_constant()
  // CHECK: arith.constant dense<1.000000e+00> : tensor<64x32xf32, #mma>
  tt.func @propagate_dot_op_to_constant() -> tensor<64x32xf32, #mma> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %cst2 = arith.constant dense<1.000000e+00> : tensor<64x32xf32, #mma>
    %0 = tt.elementwise_inline_asm "cvt.rna.tf32.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %cst : tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %2 = tt.dot %cst1, %1, %cst2 : tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
    tt.return %2 : tensor<64x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: tt.func @propagate_dot_op_to_constant_above_for()
  // CHECK: arith.constant dense<1.000000e+00> : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
  tt.func @propagate_dot_op_to_constant_above_for() -> tensor<32x128xf32, #mma> {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %loop:1 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg0 = %cst_1) -> (tensor<32x128xf32, #mma>)  : i32 {
      %0 = tt.elementwise_inline_asm "cvt.rna.tf32.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %cst : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %1 = ttg.convert_layout %0 : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %2 = ttg.convert_layout %cst_0 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %3 = tt.dot %2, %1, %arg0, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x128xf32, #mma>
      scf.yield %3 : tensor<32x128xf32, #mma>
    }
    tt.return %loop#0 : tensor<32x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // We currently don't propagate through block arguments on hoistDotOperand
  // that being said, https://github.com/triton-lang/triton/pull/5350
  // allowed to lift DotOperand(opIdx=1), which might be alright

  // CHECK: tt.func @do_not_propagate_through_block_arguments()
  // CHECK: %[[THROUGH_FOR_OP:.*]] = arith.constant dense<1.000000e+00> : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[THROUGH_FOR_OP]],
  tt.func @do_not_propagate_through_block_arguments() -> tensor<32x128xf32, #mma> {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %loop:2 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg0 = %cst, %arg1 = %cst_1) -> (tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<32x128xf32, #mma>)  : i32 {
      %0 = arith.addf %cst, %arg0 : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %1 = ttg.convert_layout %0 : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %2 = ttg.convert_layout %cst_0 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %3 = tt.dot %2, %1, %arg1, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x128xf32, #mma>
      scf.yield %0, %3 : tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<32x128xf32, #mma>
    }
    tt.return %loop#1 : tensor<32x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  tt.func @dot_op_hoisted_to_load_with_unsupported_op_and_initializer_above_slice(
                    %pa: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                    %b: tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>,
                    %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
    // CHECK: tt.func @dot_op_hoisted_to_load_with_unsupported_op_and_initializer_above_slice
    // This checks that we propagate dot op layout given the following:
    // initializer -> unsupported op -> initializer -> supported ops -> convert,
    // where initializers can be constants or loads.
    // CHECK: %[[LOAD1:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD1]]
    %offset = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %broadcast = tt.broadcast %offset : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %pa2 = tt.addptr %pa, %broadcast : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %a = tt.load %pa2 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %ae = arith.extf %a : tensor<16x16xf16, #blocked> to tensor<16x16xf32, #blocked>
    %ac = ttg.convert_layout %ae : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %ac, %b, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    tt.return %r : tensor<16x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<bf16>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = tt.fp_to_fp %[[A_DOTOP]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_CASTED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise(%pa: tensor<128x64x!tt.ptr<bf16>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a_bf16 = tt.load %pa : tensor<128x64x!tt.ptr<bf16>, #blocked>
    %a = tt.fp_to_fp %a_bf16 : tensor<128x64xbf16, #blocked> -> tensor<128x64xf16, #blocked>
    %dota = ttg.convert_layout %a: tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise_chained
//    CHECK: %[[CST_DOTOP:.*]] = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<i8>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = arith.sitofp %[[A_DOTOP]] : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> to tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_SCALED:.*]] = arith.mulf %[[A_CASTED]], %[[CST_DOTOP]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_NEGATED:.*]] = arith.negf %[[A_SCALED]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_NEGATED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise_chained(%pa: tensor<128x64x!tt.ptr<i8>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %a_i8 = tt.load %pa : tensor<128x64x!tt.ptr<i8>, #blocked>
    %a_f16 = arith.sitofp %a_i8 : tensor<128x64xi8, #blocked> to tensor<128x64xf16, #blocked>
    %a_scaled = arith.mulf %a_f16, %cst : tensor<128x64xf16, #blocked>
    %a_negated = arith.negf %a_scaled : tensor<128x64xf16, #blocked>
    %dota = ttg.convert_layout %a_negated: tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  tt.func @dot_op_hoisted_to_load_with_unsupported_op_and_initializer_above_slice(
                    %pa1: tensor<16x1x!tt.ptr<f16>, #blocked> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                    %pa2: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                    %b: tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>,
                    %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
    // CHECK: tt.func @dot_op_hoisted_to_load_with_unsupported_op_and_initializer_above_slice
    // Confirm that both loads feed directly into a convert_layout.
    // CHECK: %[[LOAD1:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD1]]
    // CHECK: %[[LOAD2:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD2]]
    %a1 = tt.load %pa1 : tensor<16x1x!tt.ptr<f16>, #blocked>
    %a2 = tt.load %pa2 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %ab = tt.broadcast %a1 : tensor<16x1xf16, #blocked> -> tensor<16x16xf16, #blocked>
    %aa = arith.addf %ab, %a2 : tensor<16x16xf16, #blocked>
    %ae = arith.extf %aa : tensor<16x16xf16, #blocked> to tensor<16x16xf32, #blocked>
    %ac = ttg.convert_layout %ae : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %ac, %b, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    tt.return %r : tensor<16x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [0, 64], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK: @remove_layout_dot_scaled
  // CHECK: %[[LOAD1:.*]] = tt.load
  // CHECK: ttg.convert_layout %[[LOAD1]]
  // CHECK: %[[LOAD2:.*]] = tt.load
  // CHECK: ttg.convert_layout %[[LOAD2]]
  // CHECK: %[[LOAD3:.*]] = tt.load
  // CHECK: ttg.convert_layout %[[LOAD3]]
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[STORE:.*]] = ttg.convert_layout
  // CHECK: tt.store %[[PTR:.+]], %[[STORE]]
  tt.func @remove_layout_dot_scaled(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0x7FC0> : tensor<32x128xbf16, #blocked>
    %cst_0 = arith.constant dense<-1> : tensor<32x4xi8, #blocked1>
    %cst_1 = arith.constant dense<7> : tensor<32x4xi16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked2>
    %cst_3 = arith.constant dense<32> : tensor<32x1xi32, #blocked3>
    %cst_4 = arith.constant dense<4> : tensor<32x1xi32, #blocked1>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %4 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi32, #blocked3>
    %6 = tt.splat %arg1 : i32 -> tensor<32x1xi32, #blocked4>
    %7 = arith.muli %3, %6 : tensor<32x1xi32, #blocked4>
    %8 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked4>
    %9 = tt.addptr %8, %7 : tensor<32x1x!tt.ptr<i8>, #blocked4>, tensor<32x1xi32, #blocked4>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %12 = tt.broadcast %9 : tensor<32x1x!tt.ptr<i8>, #blocked4> -> tensor<32x64x!tt.ptr<i8>, #blocked4>
    %13 = tt.broadcast %11 : tensor<1x64xi32, #blocked4> -> tensor<32x64xi32, #blocked4>
    %14 = tt.addptr %12, %13 : tensor<32x64x!tt.ptr<i8>, #blocked4>, tensor<32x64xi32, #blocked4>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<128x1xi32, #blocked5>
    %17 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked5>
    %18 = arith.muli %16, %17 : tensor<128x1xi32, #blocked5>
    %19 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked5>
    %20 = tt.addptr %19, %18 : tensor<128x1x!tt.ptr<i8>, #blocked5>, tensor<128x1xi32, #blocked5>
    %21 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %23 = tt.expand_dims %21 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x32xi32, #blocked5>
    %24 = tt.expand_dims %22 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %25 = tt.broadcast %20 : tensor<128x1x!tt.ptr<i8>, #blocked5> -> tensor<128x32x!tt.ptr<i8>, #blocked5>
    %26 = tt.broadcast %23 : tensor<1x32xi32, #blocked5> -> tensor<128x32xi32, #blocked5>
    %27 = tt.addptr %25, %26 : tensor<128x32x!tt.ptr<i8>, #blocked5>, tensor<128x32xi32, #blocked5>
    %28 = tt.load %14 : tensor<32x64x!tt.ptr<i8>, #blocked4>
    %29 = ttg.convert_layout %28 : tensor<32x64xi8, #blocked4> -> tensor<32x64xi8, #blocked6>
    %30 = tt.load %27 : tensor<128x32x!tt.ptr<i8>, #blocked5>
    %31 = arith.muli %4, %cst_4 : tensor<32x1xi32, #blocked1>
    %32 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked1>
    %33 = tt.addptr %32, %31 : tensor<32x1x!tt.ptr<i8>, #blocked1>, tensor<32x1xi32, #blocked1>
    %34 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x4xi32, #blocked1>
    %36 = tt.broadcast %33 : tensor<32x1x!tt.ptr<i8>, #blocked1> -> tensor<32x4x!tt.ptr<i8>, #blocked1>
    %37 = tt.broadcast %35 : tensor<1x4xi32, #blocked1> -> tensor<32x4xi32, #blocked1>
    %38 = tt.addptr %36, %37 : tensor<32x4x!tt.ptr<i8>, #blocked1>, tensor<32x4xi32, #blocked1>
    %39 = tt.load %38 : tensor<32x4x!tt.ptr<i8>, #blocked1>
    %40 = tt.bitcast %30 : tensor<128x32xi8, #blocked5> -> tensor<128x32xf8E4M3FN, #blocked5>
    %41 = ttg.convert_layout %40 : tensor<128x32xf8E4M3FN, #blocked5> -> tensor<128x32xf8E4M3FN, #blocked2>
    %42 = ttg.fp4_to_fp %29 {axis = 1 : i32} : tensor<32x64xi8, #blocked6> -> tensor<32x128xbf16, #blocked>
    %43 = arith.extui %39 : tensor<32x4xi8, #blocked1> to tensor<32x4xi16, #blocked1>
    %44 = arith.shli %43, %cst_1 : tensor<32x4xi16, #blocked1>
    %45 = tt.bitcast %44 : tensor<32x4xi16, #blocked1> -> tensor<32x4xbf16, #blocked1>
    %46 = ttg.convert_layout %45 : tensor<32x4xbf16, #blocked1> -> tensor<32x4xbf16, #ttg.slice<{dim = 2, parent = #blocked7}>>
    %47 = tt.expand_dims %46 {axis = 2 : i32} : tensor<32x4xbf16, #ttg.slice<{dim = 2, parent = #blocked7}>> -> tensor<32x4x1xbf16, #blocked7>
    %48 = tt.broadcast %47 : tensor<32x4x1xbf16, #blocked7> -> tensor<32x4x32xbf16, #blocked7>
    %49 = tt.reshape %48 : tensor<32x4x32xbf16, #blocked7> -> tensor<32x128xbf16, #linear>
    %50 = ttg.convert_layout %49 : tensor<32x128xbf16, #linear> -> tensor<32x128xbf16, #blocked>
    %51 = arith.mulf %42, %50 : tensor<32x128xbf16, #blocked>
    %52 = arith.cmpi eq, %39, %cst_0 : tensor<32x4xi8, #blocked1>
    %53 = ttg.convert_layout %52 : tensor<32x4xi1, #blocked1> -> tensor<32x4xi1, #ttg.slice<{dim = 2, parent = #blocked7}>>
    %54 = tt.expand_dims %53 {axis = 2 : i32} : tensor<32x4xi1, #ttg.slice<{dim = 2, parent = #blocked7}>> -> tensor<32x4x1xi1, #blocked7>
    %55 = tt.broadcast %54 : tensor<32x4x1xi1, #blocked7> -> tensor<32x4x32xi1, #blocked7>
    %56 = tt.reshape %55 : tensor<32x4x32xi1, #blocked7> -> tensor<32x128xi1, #linear>
    %57 = ttg.convert_layout %56 : tensor<32x128xi1, #linear> -> tensor<32x128xi1, #blocked>
    %58 = arith.select %57, %cst, %51 : tensor<32x128xi1, #blocked>, tensor<32x128xbf16, #blocked>
    %59 = ttg.convert_layout %58 : tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %60 = tt.fp_to_fp %41 : tensor<128x32xf8E4M3FN, #blocked2> -> tensor<128x32xbf16, #blocked2>
    %61 = ttg.convert_layout %60 : tensor<128x32xbf16, #blocked2> -> tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %62 = ttg.convert_layout %cst_2 : tensor<32x32xf32, #blocked2> -> tensor<32x32xf32, #mma>
    %63 = ttg.convert_layout %59 : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %64 = ttg.convert_layout %61 : tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %65 = tt.dot %63, %64, %62 : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %66 = ttg.convert_layout %65 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked2>
    %67 = ttg.convert_layout %66 : tensor<32x32xf32, #blocked2> -> tensor<32x32xf32, #blocked2>
    %68 = arith.muli %5, %cst_3 : tensor<32x1xi32, #blocked3>
    %69 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked3>
    %70 = tt.addptr %69, %68 : tensor<32x1x!tt.ptr<bf16>, #blocked3>, tensor<32x1xi32, #blocked3>
    %71 = tt.broadcast %70 : tensor<32x1x!tt.ptr<bf16>, #blocked3> -> tensor<32x32x!tt.ptr<bf16>, #blocked3>
    %72 = tt.broadcast %24 : tensor<1x32xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
    %73 = tt.addptr %71, %72 : tensor<32x32x!tt.ptr<bf16>, #blocked3>, tensor<32x32xi32, #blocked3>
    %74 = arith.truncf %67 : tensor<32x32xf32, #blocked2> to tensor<32x32xbf16, #blocked2>
    %75 = ttg.convert_layout %74 : tensor<32x32xbf16, #blocked2> -> tensor<32x32xbf16, #blocked3>
    tt.store %73, %75 : tensor<32x32x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0], [0, 1, 0], [0, 2, 0]], lane = [[0, 0, 8], [0, 0, 16], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[0, 0, 0], [16, 0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // Check that the remove-layout-conversions pass is idempotent
  // in that it keeps the convert_layout ops next to the loads
  // CHECK: tt.func @remove_layout_is_idempotent
  tt.func @remove_layout_is_idempotent(%14: tensor<32x64x!tt.ptr<i8>, #blocked2>, %39: tensor<32x4x!tt.ptr<i8>, #blocked>, %27: tensor<128x32x!tt.ptr<i8>, #blocked3>) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0x7FC0> : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_3 = arith.constant dense<7> : tensor<32x4xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %cst_4 = arith.constant dense<-1> : tensor<32x4xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    // CHECK: %[[LOAD1:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD1]]
    // CHECK: %[[LOAD2:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD2]]
    // CHECK: %[[LOAD3:.*]] = tt.load
    // CHECK: ttg.convert_layout %[[LOAD3]]
    %28 = tt.load %14 : tensor<32x64x!tt.ptr<i8>, #blocked2>
    %29 = ttg.convert_layout %28 : tensor<32x64xi8, #blocked2> -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %30 = tt.load %27 : tensor<128x32x!tt.ptr<i8>, #blocked3>
    %31 = ttg.convert_layout %30 : tensor<128x32xi8, #blocked3> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %40 = tt.load %39 : tensor<32x4x!tt.ptr<i8>, #blocked>
    %41 = ttg.convert_layout %40 : tensor<32x4xi8, #blocked> -> tensor<32x4xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    %42 = tt.bitcast %31 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %43 = ttg.fp4_to_fp %29 {axis = 1 : i32} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %44 = arith.extui %41 : tensor<32x4xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<32x4xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %45 = arith.shli %44, %cst_3 : tensor<32x4xi16, #ttg.slice<{dim = 2, parent = #linear}>>
    %46 = tt.bitcast %45 : tensor<32x4xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<32x4xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
    %47 = tt.expand_dims %46 {axis = 2 : i32} : tensor<32x4xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<32x4x1xbf16, #linear>
    %48 = tt.broadcast %47 : tensor<32x4x1xbf16, #linear> -> tensor<32x4x32xbf16, #linear>
    %49 = tt.reshape %48 : tensor<32x4x32xbf16, #linear> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %50 = arith.mulf %43, %49 : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %51 = arith.cmpi eq, %41, %cst_4 : tensor<32x4xi8, #ttg.slice<{dim = 2, parent = #linear}>>
    %52 = tt.expand_dims %51 {axis = 2 : i32} : tensor<32x4xi1, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<32x4x1xi1, #linear>
    %53 = tt.broadcast %52 : tensor<32x4x1xi1, #linear> -> tensor<32x4x32xi1, #linear>
    %54 = tt.reshape %53 : tensor<32x4x32xi1, #linear> -> tensor<32x128xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %55 = arith.select %54, %cst, %50 : tensor<32x128xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %56 = tt.fp_to_fp %42 : tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %57 = tt.dot %55, %56, %cst_0 : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return %57 : tensor<32x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  tt.func @join_reshape_dot(%112: tensor<128x32x!tt.ptr<i8>, #blocked2>, %117: tensor<128x32xi1, #blocked2>, %128: tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<16x128xf32, #mma> {
      %cst = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>
      // CHECK: %[[LOAD_I8:.*]] = tt.load {{.*}} tensor<128x32x!tt.ptr<i8>
      // CHECK: ttg.convert_layout %[[LOAD_I8]] {{.*}} #linear
      %118 = tt.load %112, %117 : tensor<128x32x!tt.ptr<i8>, #blocked2>
      %121:2 = tt.elementwise_inline_asm "" {constraints = "=r,=r,=r,=r,r", packed_element = 4 : i32, pure = true} %118 : tensor<128x32xi8, #blocked2> -> tensor<128x32xbf16, #blocked2>, tensor<128x32xbf16, #blocked2>
      %122 = tt.join %121#0, %121#1 : tensor<128x32xbf16, #blocked2> -> tensor<128x32x2xbf16, #blocked4>
      %123 = tt.reshape %122 : tensor<128x32x2xbf16, #blocked4> -> tensor<128x64xbf16, #blocked5>
      %124 = tt.trans %123 {order = array<i32: 1, 0>} : tensor<128x64xbf16, #blocked5> -> tensor<64x128xbf16, #blocked6>
      %126 = ttg.convert_layout %124 : tensor<64x128xbf16, #blocked6> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %127 = ttg.convert_layout %cst : tensor<16x128xf32, #blocked> -> tensor<16x128xf32, #mma>
      %129 = ttg.convert_layout %126 : tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %130 = tt.dot %128, %129, %127, inputPrecision = tf32 : tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x128xf32, #mma>
      tt.return %130 : tensor<16x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0]], warp = [], block = []}>
module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  // CHECK-LABEL: join_forward
  tt.func @join_forward(%arg0: tensor<2x16xf32, #linear>) -> tensor<2x16x2xf32, #blocked> {
    // CHECK-LABEL: tt.join
    // CHECK-LABEL: ttg.convert_layout
    %0 = ttg.convert_layout %arg0 : tensor<2x16xf32, #linear> -> tensor<2x16xf32, #blocked1>
    %1 = tt.join %0, %0 : tensor<2x16xf32, #blocked1> -> tensor<2x16x2xf32, #blocked>
    tt.return %1 : tensor<2x16x2xf32, #blocked>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
// CHECK: [[$BLOCK:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_no_redundant_convert_layout
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_no_redundant_convert_layout(
        %arg0: tensor<128x128xf8E4M3FN, #dot_op_a>,
        %arg1: tensor<128x128xf8E4M3FN, #dot_op_b>,
        %arg2: tensor<128x4xi8, #linear>,
        %arg3: tensor<128x4xi8, #linear1>,
        %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // CHECK: %[[RET:.+]] = scf.for
    // CHECK-NEXT: %[[DOT_RET:.+]] = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false}
    // CHECK-NEXT: scf.yield %[[DOT_RET]]
    // CHECK-NEXT: }
    // CHECK-NEXT: ttg.convert_layout %[[RET]] : tensor<128x128xf32, #mma> -> tensor<128x128xf32, [[$BLOCK]]>
    // CHECK-NEXT: tt.store
    %1 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst0) -> (tensor<128x128xf32, #blocked1>) {
      %4 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #dot_op_a>, tensor<128x4xi8, #linear> * tensor<128x128xf8E4M3FN, #dot_op_b>, tensor<128x4xi8, #linear1> -> tensor<128x128xf32, #mma>
      %5 = ttg.convert_layout %4 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
      scf.yield %5 : tensor<128x128xf32, #blocked1>
    }
    %7 = ttg.convert_layout %1 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %7 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
