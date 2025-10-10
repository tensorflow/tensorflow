// RUN: xla-opt %s -split-input-file \
// RUN: -tensor-lower-to-triton \
// RUN: | FileCheck %s

// CHECK: func @lower_bitcast(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<2x4x8xi32>
func.func @lower_bitcast(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xi32> {
  // CHECK: %[[RES:.*]] = tt.bitcast %[[ARG]] : tensor<2x4x8xf32> -> tensor<2x4x8xi32>
  %0 = tensor.bitcast %arg0 : tensor<2x4x8xf32> to tensor<2x4x8xi32>
  // CHECK: return %[[RES]] : tensor<2x4x8xi32>
  return %0 : tensor<2x4x8xi32>
}

// CHECK: func @lower_splat(%[[ARG:.*]]: f32) -> tensor<16xf32>
func.func @lower_splat(%arg0: f32) -> tensor<16xf32> {
  // CHECK: %[[RES:.*]] = tt.splat %[[ARG]] : f32 -> tensor<16xf32>
  %0 = tensor.splat %arg0 : tensor<16xf32>
  // CHECK: return %[[RES]] : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK: func @lower_splat_falls_back_to_tensor_splat_for_dynamic_shape(%[[ARG:.*]]: f32) -> tensor<?x16xf32>
func.func @lower_splat_falls_back_to_tensor_splat_for_dynamic_shape(%arg0: f32) -> tensor<?x16xf32> {
  // CHECK: %[[DIM_CONST:.*]] = arith.constant 36 : index
  %dim = arith.constant 36 : index
  // CHECK: %[[RES:.*]] = tensor.splat %[[ARG]][%[[DIM_CONST]]] : tensor<?x16xf32>
  %0 = tensor.splat %arg0[%dim] : tensor<?x16xf32>
  // CHECK: return %[[RES]] : tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}