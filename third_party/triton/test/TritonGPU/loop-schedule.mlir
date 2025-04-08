// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-assign-latencies=num-stages=3 -tritongpu-test-pipeline-schedule-loop | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#ALs0 = #ttg.slice<{parent=#AL, dim=0}>
#BLs0 = #ttg.slice<{parent=#BL, dim=0}>
#CLs0 = #ttg.slice<{parent=#C, dim=0}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABLE: @matmul_loop_load_acc
// CHECK: tt.load %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
// CHECK: tt.load %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
// CHECK: tt.load %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
tt.func @matmul_loop_load_acc(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %C : !tt.ptr<f32> {tt.divisibility = 16 : i32},
                  %c_init: tensor<128x128xf32, #C>) -> tensor<128x128xf32, #C> {

  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
  // C ptrs
  %c_ptr_splat = tt.splat %C : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #C>
  %c_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #CLs0>
  %c_tmp1 = tt.expand_dims %c_tmp0 {axis = 0 : i32} : tensor<128xi32, #CLs0> -> tensor<1x128xi32, #C>
  %c_offs = tt.broadcast %c_tmp1 : tensor<1x128xi32, #C> -> tensor<128x128xi32, #C>
  %c_ptr_init = tt.addptr %c_ptr_splat, %c_offs : tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xi32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c_off = arith.constant dense<4> : tensor<128x128xi32, #C>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %c_ptr = %c_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c_ = tt.load %c_ptr : tensor<128x128x!tt.ptr<f32>, #C>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_c_ptr = tt.addptr %c_ptr, %c_off : tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xi32, #C>
    scf.yield %next_a_ptr, %next_b_ptr, %next_c_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @fused_loop
tt.func public @fused_loop(%arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
  %c10_i32 = arith.constant 10 : i32
  %false = arith.constant false
  %0 = ub.poison : !tt.tensordesc<tensor<64x256xf16>>
  %cst = arith.constant dense<0> : tensor<128x1xi64, #blocked>
  %c-1_i32 = arith.constant -1 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>

  %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
  %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
  %3 = arith.extsi %arg7 : i32 to i64
  %4 = tt.make_tensor_descriptor %arg5, [%arg7, %arg7], [%3, %c1_i64] : <f16>, <tensor<64x256xf16>>
  %5 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
  %7 = tt.splat %3 : i64 -> tensor<128x1xi64, #blocked>

  // CHECK: scf.for
  %8:9 = scf.for %arg29 = %c0_i32 to %arg7 step %c1_i32 iter_args(%arg30 = %c-1_i32, %arg31 = %4, %arg32 = %c0_i32, %arg33 = %arg5, %arg34 = %cst_0, %arg35 = %c0_i32, %arg36 = %cst, %arg37 = %0, %arg38 = %false) -> (i32, !tt.tensordesc<tensor<64x256xf16>>, i32, !tt.ptr<f16>, tensor<128x256xf32, #mma>, i32, tensor<128x1xi64, #blocked>, !tt.tensordesc<tensor<64x256xf16>>, i1)  : i32 {
    %9 = arith.addi %arg30, %c1_i32 : i32
    %10 = arith.cmpi eq, %arg30, %c10_i32 : i32
    %11 = arith.select %10, %c0_i32, %9 : i32
    %12 = arith.cmpi eq, %11, %c0_i32 : i32

    // This op is a distance 1 dependency of itself.
    // CHECK: {_test_marker_0, loop.cluster = 4 : i32, loop.stage = 0 : i32}
    %13 = arith.select %12, %c0_i32, %arg32 {_test_marker_0} : i32

    %14 = arith.select %12, %arg31, %arg37 : !tt.tensordesc<tensor<64x256xf16>>
    %15 = arith.select %12, %c10_i32, %arg35 : i32
    %16 = scf.if %12 -> (tensor<128x1xi64, #blocked>) {
      %32 = arith.muli %cst, %7 : tensor<128x1xi64, #blocked>
      scf.yield %32 : tensor<128x1xi64, #blocked>
    } else {
      scf.yield %arg36 : tensor<128x1xi64, #blocked>
    }
    %17 = tt.splat %arg33 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %18 = tt.addptr %17, %16 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi64, #blocked>
    %19 = tt.broadcast %18 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %20 = tt.addptr %19, %5 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %21 = tt.addptr %arg33, %c64_i32 : !tt.ptr<f16>, i32
    %22 = tt.load %20 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %23 = ttg.local_alloc %22 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %24 = arith.muli %13, %c64_i32 : i32
    %25 = tt.descriptor_load %14[%24, %15] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
    %26 = ttg.local_alloc %25 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %27 = ttng.warp_group_dot %23, %26, %arg34, %arg38 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %28 = arith.addi %13, %c1_i32 : i32

    // This op is in the backward slice of `_test_marker_2` and the epilogue.
    // CHECK: {_test_marker_1, loop.cluster = 3 : i32, loop.stage = 1 : i32}
    %29 = arith.cmpi eq, %11, %c10_i32 {_test_marker_1} : i32

    // CHECK: {_test_marker_2, loop.cluster = 3 : i32, loop.stage = 1 : i32}
    %30 = arith.select %29, %arg5, %21 {_test_marker_2} : !tt.ptr<f16>

    %31 = arith.cmpi ne, %11, %c10_i32 : i32

    scf.if %29 {
      "use"(%27) : (tensor<128x256xf32, #mma>) -> ()
      // CHECK: {_test_marker_3, loop.cluster = 5 : i32, loop.stage = 2 : i32}
    } {_test_marker_3}
    scf.yield %11, %14, %28, %30, %27, %15, %16, %14, %31 : i32, !tt.tensordesc<tensor<64x256xf16>>, i32, !tt.ptr<f16>, tensor<128x256xf32, #mma>, i32, tensor<128x1xi64, #blocked>, !tt.tensordesc<tensor<64x256xf16>>, i1
  }
  tt.return
}

}

// -----

// CHECK-LABEL: @prologue_backward_slice
tt.func @prologue_backward_slice(%ub: i32, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK: scf.for
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK: scf.if
    %0 = scf.if %cond -> i32 {
      scf.yield %c0_i32 : i32
    } else {
      scf.yield %c1_i32 : i32
    }
    // CHECK: loop.cluster = 0 : i32, loop.stage = 0 : i32

    // CHECK: op.with_region
    %1 = "op.with_region"() ({
      "use"(%0) : (i32) -> ()
    }) : () -> i32
    // CHECK: loop.cluster = 1 : i32, loop.stage = 0 : i32

    // CHECK: op.with_region
    "op.with_region"() ({
      "use"(%1) : (i32) -> ()
    }) {tt_latency = 2 : i32} : () -> ()
    // CHECK: loop.cluster = 1 : i32, loop.stage = 0 : i32

  } {tt.num_stages = 3 : i32}

  tt.return
}

// -----

// CHECK-LABEL: @epilogue_forward_slice
tt.func @epilogue_forward_slice(%ub: i32, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK: scf.for
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK: "latency.op"() {loop.cluster = 3 : i32, loop.stage = 0 : i32
    %0 = "latency.op"() {tt_latency = 2 : i32} : () -> i32
    // CHECK: scf.if
    %1 = scf.if %cond -> i32 {
      scf.yield %0 : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    // CHECK: {loop.cluster = 1 : i32, loop.stage = 2 : i32}

    // CHECK: "use"(%{{.*}}) {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    "use"(%1) : (i32) -> ()

  } {tt.num_stages = 3 : i32}

  tt.return
}

// -----

// CHECK-LABEL: @prologue_latency
tt.func @prologue_latency(%ub: i32, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK: scf.for
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK: "some.op"() {loop.cluster = 0 : i32, loop.stage = 0 : i32}
    %0 = "some.op"() : () -> i32
    // CHECK: scf.if
    %1 = scf.if %cond -> i32 {
      scf.yield %0 : i32
    } else {
      scf.yield %c0_i32 : i32
    } {tt_latency = 2 : i32}
    // CHECK: loop.cluster = 0 : i32, loop.stage = 0 : i32

  } {tt.num_stages = 3 : i32}

  tt.return
}
