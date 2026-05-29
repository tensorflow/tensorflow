// RUN: xla-opt %s -split-input-file \
// RUN: -xtile-lower-to-triton \
// RUN: -triton-xla-fold-reshape-around-for-loop \
// RUN: -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK: func @lower_dot_scaled_add_to_triton(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>, %[[ACC:.*]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_add_to_triton(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>,
  %acc: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = tt.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]], %[[ACC]] lhs = e5m2 rhs = e5m2 {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK-NOT: arith.addf
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>,
     fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %1 = arith.addf %acc, %0 : tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// CHECK-LABEL: func @lower_dot_scaled_in_loop_non_canonical
func.func @lower_dot_scaled_in_loop_non_canonical(
  %lhs: tensor<1x128x128xf8E5M2>, %lhs_scale: tensor<1x128x4xi8>,
  %rhs: tensor<1x128x256xf8E5M2>, %rhs_scale: tensor<1x256x4xi8>,
  %acc: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[INIT_R:.*]] = tt.reshape %arg4 : tensor<1x128x256xf32> -> tensor<128x256xf32>
  // CHECK: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC_2D:.*]] = %[[INIT_R]]) -> (tensor<128x256xf32>) {
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%accum = %acc) -> tensor<1x128x256xf32> {
    // CHECK: %[[DOT:.*]] = tt.dot_scaled %{{.*}} scale %{{.*}}, %{{.*}} scale %{{.*}}, %[[ACC_2D]]
    // CHECK: scf.yield %[[DOT]] : tensor<128x256xf32>
    %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
      {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
       fastMath = true} : tensor<1x128x128xf8E5M2>, tensor<1x128x4xi8> * tensor<1x128x256xf8E5M2>, tensor<1x256x4xi8> -> tensor<1x128x256xf32>
    %1 = arith.addf %accum, %0 : tensor<1x128x256xf32>
    scf.yield %1 : tensor<1x128x256xf32>
  }
  // CHECK: %[[FINAL:.*]] = tt.reshape %[[LOOP]] : tensor<128x256xf32> -> tensor<1x128x256xf32>
  // CHECK: return %[[FINAL]]
  return %res : tensor<1x128x256xf32>
}

// CHECK: func @lower_dot_scaled_without_add_falls_back_to_xtile(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_without_add_falls_back_to_xtile(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>)
  -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = xtile.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]] {{.*}}fastMath = true{{.*}} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>,
     fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}