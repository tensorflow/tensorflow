// RUN: xla-opt %s -split-input-file \
// RUN: -stablehlo-lower-to-triton \
// RUN: | FileCheck %s

// CHECK: func @lower_transpose(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
func.func @lower_transpose(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %[[RES:.*]] = tt.trans %[[ARG]] {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  // CHECK: return %[[RES]] : tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

// CHECK: func @lower_iota_to_make_range() -> tensor<16xi32>
func.func @lower_iota_to_make_range() -> tensor<16xi32> {
  // CHECK: %[[RES:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %0 = stablehlo.iota dim = 0 : tensor<16xi32>
  // CHECK: return %[[RES]] : tensor<16xi32>
  return %0 : tensor<16xi32>
}

// CHECK: func @lower_iota_on_multidimensional_tensor_falls_back_to_stablehlo() -> tensor<16x32xi32>
func.func @lower_iota_on_multidimensional_tensor_falls_back_to_stablehlo() -> tensor<16x32xi32> {
  // CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<16x32xi32>
  %0 = stablehlo.iota dim = 0 : tensor<16x32xi32>
  // CHECK: return %[[RES]] : tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK: func @lower_iota_on_non_signed_32_bit_tensor_falls_back_to_stablehlo() -> tensor<8xui32>
func.func @lower_iota_on_non_signed_32_bit_tensor_falls_back_to_stablehlo() -> tensor<8xui32> {
  // CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<8xui32>
  %0 = stablehlo.iota dim = 0 : tensor<8xui32>
  // CHECK: return %[[RES]] : tensor<8xui32>
  return %0 : tensor<8xui32>
}

// CHECK: func @lower_broadcast_in_dim(%[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<8x2x4x16xf32>
func.func @lower_broadcast_in_dim(%arg0: tensor<2x4xf32>) -> tensor<8x2x4x16xf32> {
  // CHECK: %[[RES_EXPAND_DIMS_0:.*]] = tt.expand_dims %[[ARG0]] {axis = 0 : i32} : tensor<2x4xf32> -> tensor<1x2x4xf32>
  // CHECK: %[[RES_EXPAND_DIMS_1:.*]] = tt.expand_dims %[[RES_EXPAND_DIMS_0]] {axis = 3 : i32} : tensor<1x2x4xf32> -> tensor<1x2x4x1xf32>
  // CHECK: %[[RES:.*]] = tt.broadcast %[[RES_EXPAND_DIMS_1]] : tensor<1x2x4x1xf32> -> tensor<8x2x4x16xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x4xf32>) -> tensor<8x2x4x16xf32>
  // CHECK: return %[[RES]] : tensor<8x2x4x16xf32>
  return %0 : tensor<8x2x4x16xf32>
}

// CHECK: func @lower_broadcast_in_dim_on_0d_tensor_produced_by_to_tensor_to_splat(%[[ARG0:.*]]: f32) -> tensor<4x2xf32>
func.func @lower_broadcast_in_dim_on_0d_tensor_produced_by_to_tensor_to_splat(%arg0: f32) -> tensor<4x2xf32> {
  // CHECK-NOT: xtile.to_tensor
  // CHECK: %[[RES:.*]] = tt.splat %[[ARG0]] : f32 -> tensor<4x2xf32>
  %to_tensor = xtile.to_tensor %arg0 : f32
  %0 = stablehlo.broadcast_in_dim %to_tensor, dims = [] : (tensor<f32>) -> tensor<4x2xf32>
  // CHECK: return %[[RES]] : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK: func @reduce(%[[ARG0:.*]]: tensor<16x8xf32>) -> tensor<8xf32>
func.func @reduce(%arg0: tensor<16x8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RES:.*]] = "tt.reduce"(%[[ARG0]]) <{axis = 0 : i32}> ({
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  //CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: %[[RES:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
    // CHECK: tt.reduce.return %[[RES]] : f32
    %extracted_arg1 = xtile.to_scalar %arg1 : tensor<f32>
    %extracted_arg2 = xtile.to_scalar %arg2 : tensor<f32>
    %2 = arith.addf %extracted_arg1, %extracted_arg2 : f32
    %3 = xtile.to_tensor %2 : f32
    stablehlo.return %3 : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK: func @reduce_to_scalar_followed_by_extract(%[[ARG0:.*]]: tensor<16xf32>) -> f32
func.func @reduce_to_scalar_followed_by_extract(%arg0: tensor<16xf32>) -> f32 {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[ARG0]]) <{axis = 0 : i32}> ({
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  //CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: %[[RES:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
    // CHECK: tt.reduce.return %[[RES]] : f32
    %extracted_arg1 = xtile.to_scalar %arg1 : tensor<f32>
    %extracted_arg2 = xtile.to_scalar %arg2 : tensor<f32>
    %2 = arith.addf %extracted_arg1, %extracted_arg2 : f32
    %3 = xtile.to_tensor %2 : f32
    stablehlo.return %3 : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  // CHECK-NOT: xtile.to_tensor
  // CHECK-NOT: xtile.to_scalar
  %extract = xtile.to_scalar %1 : tensor<f32>
  // CHECK: return %[[REDUCE_RESULT:.*]] : f32
  return %extract : f32
}

// CHECK: func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%[[ARG0:.*]]: tensor<16x8x4xf32>) -> tensor<4xf32>
func.func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%arg0: tensor<16x8x4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %{{.*}}) across dimensions = [0, 1] : (tensor<16x8x4xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %extracted_arg1 = xtile.to_scalar %arg1 : tensor<f32>
    %extracted_arg2 = xtile.to_scalar %arg2 : tensor<f32>
    %2 = arith.addf %extracted_arg1, %extracted_arg2 : f32
    %3 = xtile.to_tensor %2 : f32
    stablehlo.return %3 : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<16x8x4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: return %[[RES]] : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK: func @reduce_with_multiple_inputs(%[[ARG0:.*]]: tensor<16x8xf32>, %[[ARG1:.*]]: tensor<16x8xf32>) -> tensor<8xf32>
func.func @reduce_with_multiple_inputs(%arg0: tensor<16x8xf32>, %arg1: tensor<16x8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[ARG0]], %[[ARG1]]) <{axis = 0 : i32}> ({
  %1, %2 = "stablehlo.reduce"(%arg0, %arg1, %0, %0) ({
  ^bb0(%arg0_reducer: tensor<f32>, %arg1_reducer: tensor<f32>, %arg2_reducer: tensor<f32>, %arg3_reducer: tensor<f32>):
    %extracted_arg0 = xtile.to_scalar %arg0_reducer : tensor<f32>
    %extracted_arg1 = xtile.to_scalar %arg1_reducer : tensor<f32>
    %2 = arith.addf %extracted_arg0, %extracted_arg1 : f32
    %3 = xtile.to_tensor %2 : f32
    %extracted_arg2 = xtile.to_scalar %arg2_reducer : tensor<f32>
    %extracted_arg3 = xtile.to_scalar %arg3_reducer : tensor<f32>
    %4 = arith.addf %extracted_arg2, %extracted_arg3 : f32
    %5 = xtile.to_tensor %4 : f32
    stablehlo.return %3, %5 : tensor<f32>, tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16x8xf32>, tensor<16x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %1 : tensor<8xf32>
}

func.func @lower_reshape(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %[[RES:.*]] = tt.reshape %[[ARG]] : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

