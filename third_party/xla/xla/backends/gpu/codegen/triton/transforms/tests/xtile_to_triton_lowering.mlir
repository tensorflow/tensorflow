// RUN: xla-opt %s -split-input-file \
// RUN: -xtile-lower-to-triton \
// RUN: | FileCheck %s


// CHECK: func @lower_dot_scaled_add_to_triton(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>, %[[ACC:.*]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_add_to_triton(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>,
  %acc: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = tt.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]], %[[ACC]] lhs = e5m2 rhs = e5m2 {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK-NOT: arith.addf
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %1 = arith.addf %acc, %0 : tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// CHECK: func @lower_dot_scaled_without_add_falls_back_to_xtile(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_without_add_falls_back_to_xtile(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>)
  -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = xtile.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]] {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}