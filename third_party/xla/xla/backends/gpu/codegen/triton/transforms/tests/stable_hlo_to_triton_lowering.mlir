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
  // CHECK-NOT: tensor.from_elements
  // CHECK: %[[RES:.*]] = tt.splat %[[ARG0]] : f32 -> tensor<4x2xf32>
  %to_tensor = tensor.from_elements %arg0 : tensor<f32>
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
    // CHECK: %[[ARG1_CAST:.*]] = tensor.from_elements %[[ARG1]] : tensor<f32>
    // CHECK: %[[ARG2_CAST:.*]] = tensor.from_elements %[[ARG2]] : tensor<f32>
    // CHECK: %[[RES:.*]] = arith.addf %[[ARG1_CAST]], %[[ARG2_CAST]] : tensor<f32>
    // CHECK: %[[RES_CAST:.*]] = tensor.extract %[[RES]][] : tensor<f32>
    // CHECK: tt.reduce.return %[[RES_CAST]] : f32
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
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
    // CHECK: %[[RES:.*]] = arith.addf {{.*}} : tensor<f32>
    // CHECK: tt.reduce.return {{.*}} : f32
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  // CHECK-NOT: tensor.from_elements
  // CHECK-NOT: tensor.extract
  %extract = tensor.extract %1[] : tensor<f32>
  // CHECK: return %[[REDUCE_RESULT:.*]] : f32
  return %extract : f32
}

// CHECK: func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%[[ARG0:.*]]: tensor<16x8x4xf32>) -> tensor<4xf32>
func.func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%arg0: tensor<16x8x4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %{{.*}}) across dimensions = [0, 1] : (tensor<16x8x4xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
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
    %add0 = arith.addf %arg0_reducer, %arg1_reducer : tensor<f32>
    %add1 = arith.addf %arg2_reducer, %arg3_reducer : tensor<f32>
    stablehlo.return %add0, %add1 : tensor<f32>, tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16x8xf32>, tensor<16x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %1 : tensor<8xf32>
}

func.func @lower_reshape(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %[[RES:.*]] = tt.reshape %[[ARG]] : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

// CHECK-LABEL: @reshape_0d_to_0d_folds(%arg0: tensor<f32>)
func.func @reshape_0d_to_0d_folds(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @reshape_0d_to_2d_splats(%arg0: tensor<f32>)
func.func @reshape_0d_to_2d_splats(%arg0: tensor<f32>) -> tensor<1x1xf32> {
  // CHECK: %[[SCALAR:.*]] = tensor.extract %arg0[] : tensor<f32>
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[SCALAR]] : f32 -> tensor<1x1xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<1x1xf32>
  // CHECK: return %[[SPLAT]]
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: @reshape_2d_to_0d_reduces(%arg0: tensor<1x1xf32>)
func.func @reshape_2d_to_0d_reduces(%arg0: tensor<1x1xf32>) -> tensor<f32> {
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %arg0 allow_reorder : tensor<1x1xf32> -> tensor<1xf32>
  // CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[RESHAPE]]) <{axis = 0 : i32}> ({
  // CHECK:  ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:    %[[ADD:.*]] = arith.addf %arg1, %arg2 : f32
  // CHECK:    tt.reduce.return %[[ADD]] : f32
  // CHECK:  }) : (tensor<1xf32>) -> f32
  // CHECK:  %[[REDUCE_TENSOR:.*]] = tensor.from_elements %[[REDUCE]] : tensor<f32>
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<f32>
  // CHECK: return %[[REDUCE_TENSOR]]
  return %0 : tensor<f32>
}
