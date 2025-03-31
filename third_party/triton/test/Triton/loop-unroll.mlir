// RUN: triton-opt --split-input-file %s -triton-loop-unroll | FileCheck %s

tt.func @add_kernel_unroll(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is unrolled by factor of 2 and is followed by a reminder loop.
  // CHECK-LABEL: add_kernel_unroll
  // CHECK: scf.for
  // CHECK-COUNT-2: tt.load
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK-NOT: tt.load
  // CHECK: tt.num_stages = 1 : i32
  %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %4 = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  } {tt.loop_unroll_factor = 2 : i32}
  tt.return
}

// -----

tt.func @add_kernel_nounroll(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is not unrolled.
  // CHECK-LABEL: add_kernel_nounroll
  // CHECK: scf.for
  // CHECK-COUNT-1: tt.load
  // CHECK-NOT: tt.load
  // CHECK-NOT: scf.for
  %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %4 = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
