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
