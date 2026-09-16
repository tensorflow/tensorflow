// RUN: xla-opt %s --triton-xla-fold-reshape-around-for-loop | FileCheck %s

// CHECK-LABEL: func @fold_reshape_around_for_loop
func.func @fold_reshape_around_for_loop(%arg0: index, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // CHECK: %[[LOOP:.*]] = scf.for
  %loop = scf.for %i = %arg0 to %arg0 step %arg0 iter_args(%arg = %arg1) -> (tensor<1x16xf32>) {
    %reshaped = tt.reshape %arg : tensor<1x16xf32> -> tensor<16xf32>
    %back = tt.reshape %reshaped : tensor<16xf32> -> tensor<1x16xf32>
    // CHECK: scf.yield %{{.*}} : tensor<16xf32>
    scf.yield %back : tensor<1x16xf32>
  }
  // CHECK: %[[RES:.*]] = tt.reshape %[[LOOP]]
  // CHECK: return %[[RES]] : tensor<1x16xf32>
  return %loop : tensor<1x16xf32>
}

// CHECK-LABEL: func @skip_non_rank_increasing_reshape_around_for_loop
func.func @skip_non_rank_increasing_reshape_around_for_loop(%arg0: index, %arg1: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // CHECK: %[[LOOP:.*]] = scf.for
  %loop = scf.for %i = %arg0 to %arg0 step %arg0 iter_args(%arg = %arg1) -> (tensor<2x8xf32>) {
    %reshaped = tt.reshape %arg : tensor<2x8xf32> -> tensor<4x4xf32>
    %back = tt.reshape %reshaped : tensor<4x4xf32> -> tensor<2x8xf32>
    scf.yield %back : tensor<2x8xf32>
  }
  // CHECK: return %[[LOOP]]
  return %loop : tensor<2x8xf32>
}
