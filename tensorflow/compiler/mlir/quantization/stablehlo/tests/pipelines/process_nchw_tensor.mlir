// RUN: stablehlo-quant-opt %s -stablehlo-process-nchw-tensor \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// Tests that a `convolution(%activation, %weight)` with the activation tensor
// NCHW format is converted to NHWC convolution. Transpose ops are inserted to
// the activation and output to match the function signature. The weight
// constant is transposed.

// CHECK-LABEL: nchw_conv
// CHECK-SAME: %[[ARG:.+]]: tensor<1x8x4x4xf32>
func.func @nchw_conv(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %0 = stablehlo.constant() {value = dense<7.000000e+00> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x4x4xf32>, tensor<8x8x3x3xf32>) -> tensor<1x8x4x4xf32>
  return %2 : tensor<1x8x4x4xf32>
}
// CHECK-DAG: %[[CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x8x8xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x8x4x4xf32>) -> tensor<1x4x4x8xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x4x8xf32>, tensor<3x3x8x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[CONV]], dims = [0, 3, 1, 2] : (tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32>
// CHECK: return %[[TRANSPOSE_1]]

// -----

// Tests that a `add(convolution(%activation, %weight), %bias)` with the
// activation tensor of NCHW format is converted to NHWC convolution + add
// operation. Transpose ops are inserted to activations and outputs to match the
// function signature. Constants are also transposed accordingly.

// CHECK-LABEL: nchw_conv_with_bias_add
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x5x5xf32>
func.func @nchw_conv_with_bias_add(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x4x5x5xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4x2x3x3xf32>
  %1 = stablehlo.constant dense<3.000000e+00> : tensor<1x4x5x5xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  %3 = stablehlo.add %2, %1 : tensor<1x4x5x5xf32>
  return %3 : tensor<1x4x5x5xf32>
}
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x2x4xf32>
// CHECK-DAG: %[[BIAS_CONST:.+]] = stablehlo.constant {{.*}} : tensor<1x5x5x4xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x2x5x5xf32>) -> tensor<1x5x5x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[WEIGHT_CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %[[CONV]], %[[BIAS_CONST]] : tensor<1x5x5x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[ADD]], dims = [0, 3, 1, 2] : (tensor<1x5x5x4xf32>) -> tensor<1x4x5x5xf32>
// CHECK: return %[[TRANSPOSE_1]]

// -----

// Tests that a `add(convolution(%activation, %weight), %bias)` pattern with the
// activation tensor of NCHW format and non-constant bias is converted to NHWC
// convolution, but without the deferred transpose for `stablehlo.add`.
// Transpose ops are inserted to the activation and output of
// `stablehlo.convolution`. The weight constants is transposed.

// CHECK-LABEL: nchw_conv_with_nonconst_bias_add
// CHECK-SAME: %[[ARG_0:.+]]: tensor<1x2x5x5xf32>
// CHECK-SAME: %[[ARG_1:.+]]: tensor<1x4x5x5xf32>
func.func @nchw_conv_with_nonconst_bias_add(%arg0: tensor<1x2x5x5xf32>, %arg1: tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4x2x3x3xf32>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  %2 = stablehlo.add %1, %arg1 : tensor<1x4x5x5xf32>
  return %2 : tensor<1x4x5x5xf32>
}
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x2x4xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG_0]], dims = [0, 2, 3, 1] : (tensor<1x2x5x5xf32>) -> tensor<1x5x5x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[WEIGHT_CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[CONV]], dims = [0, 3, 1, 2] : (tensor<1x5x5x4xf32>) -> tensor<1x4x5x5xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %[[TRANSPOSE_1]], %[[ARG_1]] : tensor<1x4x5x5xf32>
// CHECK: return %[[ADD]]

// -----

// Tests that a `reduce_window{max}(add(convolution(%activation, %weight), %bias), %init_value)`
// with the activation tensor of NCHW format is converted to NHWC convolution +
// add + reduce_window (with max) operation. Transpose ops are inserted to
// activation and the final result to match the function signature. Constants
// are also transposed accordingly.

// CHECK-LABEL: nchw_conv_with_bias_add_max_pool
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x5x5xf32>
func.func @nchw_conv_with_bias_add_max_pool(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x4x2x2xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4x2x3x3xf32>
  %1 = stablehlo.constant dense<3.000000e+00> : tensor<1x4x5x5xf32>
  %5 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  %3 = stablehlo.add %2, %1 : tensor<1x4x5x5xf32>
  %4 = "stablehlo.reduce_window"(%3, %5) ({  // max pool
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 2, 2>
  } : (tensor<1x4x5x5xf32>, tensor<f32>) -> tensor<1x4x2x2xf32>
  return %4 : tensor<1x4x2x2xf32>
}
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x2x4xf32>
// CHECK-DAG: %[[BIAS_CONST:.+]] = stablehlo.constant {{.*}} : tensor<1x5x5x4xf32>
// CHECK-DAG: %[[INIT_VALUE_CONST:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x2x5x5xf32>) -> tensor<1x5x5x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[WEIGHT_CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %[[CONV]], %[[BIAS_CONST]] : tensor<1x5x5x4xf32>
// CHECK: %[[REDUCE_WINDOW_MAX:.+]] = "stablehlo.reduce_window"(%[[ADD]], %[[INIT_VALUE_CONST:.+]])
// CHECK: stablehlo.maximum
// CHECK: {window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x5x5x4xf32>, tensor<f32>) -> tensor<1x2x2x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[REDUCE_WINDOW_MAX]], dims = [0, 3, 1, 2] : (tensor<1x2x2x4xf32>) -> tensor<1x4x2x2xf32>
// CHECK: return %[[TRANSPOSE_1]]

// -----

// Tests that a `maximum(add(convolution(%activation, %weight), %bias), %zero)`
// with the activation tensor of NCHW format is converted to NHWC convolution +
// add + maximum operation. Transpose ops are inserted to the activation and the
// final output to match the function signature. Constants are also transpose-
// folded accordingly.

// CHECK-LABEL: nchw_conv_with_bias_add_relu
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x5x5xf32>
func.func @nchw_conv_with_bias_add_relu(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x4x5x5xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4x2x3x3xf32>
  %5 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x5x5xf32>
  %1 = stablehlo.constant dense<3.000000e+00> : tensor<1x4x5x5xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  %3 = stablehlo.add %2, %1 : tensor<1x4x5x5xf32>
  %4 = stablehlo.maximum %3, %5 : tensor<1x4x5x5xf32>
  return %4 : tensor<1x4x5x5xf32>
}
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x2x4xf32>
// CHECK-DAG: %[[ZERO_CONST:.+]] = stablehlo.constant {{.*}} : tensor<1x5x5x4xf32>
// CHECK-DAG: %[[BIAS_CONST:.+]] = stablehlo.constant {{.*}} : tensor<1x5x5x4xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x2x5x5xf32>) -> tensor<1x5x5x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[WEIGHT_CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %[[CONV]], %[[BIAS_CONST]] : tensor<1x5x5x4xf32>
// CHECK: %[[MAX:.+]] = stablehlo.maximum %[[ADD]], %[[ZERO_CONST]] : tensor<1x5x5x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[MAX]], dims = [0, 3, 1, 2] : (tensor<1x5x5x4xf32>) -> tensor<1x4x5x5xf32>
// CHECK: return %[[TRANSPOSE_1]]

// -----

// Tests that a `maximum(add(convolution(%activation, %weight), broadcast(%bias)
// ), %zero)` with the activation tensor of NCHW format is converted to NHWC
// convolution + add + maximum operation. Transpose ops are inserted to the
// first activation, final output, and the bias constant (after the broadcast),
// to match the function signature. Constants are also transpose-folded
// accordingly.
//
// Note that the `transpose` after the `broadcast_in_dim` is not folded by the
// `FoldConstantTransposePass`.

// CHECK-LABEL: nchw_conv_with_broadcasted_bias_add_relu
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x5x5xf32>
func.func @nchw_conv_with_broadcasted_bias_add_relu(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x4x5x5xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<4x2x3x3xf32>  // weight
  %1 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>  // bias
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x5x5xf32>  // relu
  %3 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<1x4x5x5xf32>
  %4 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  %5 = stablehlo.add %4, %3 : tensor<1x4x5x5xf32>
  %6 = stablehlo.maximum %5, %2 : tensor<1x4x5x5xf32>
  return %6 : tensor<1x4x5x5xf32>
}
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} : tensor<3x3x2x4xf32>
// CHECK-DAG: %[[ZERO_CONST:.+]] = stablehlo.constant {{.*}} : tensor<1x5x5x4xf32>
// CHECK-DAG: %[[BIAS_CONST:.+]] = stablehlo.constant {{.*}} : tensor<4xf32>
// CHECK-DAG: %[[BROADCAST_IN_DIM:.+]] = stablehlo.broadcast_in_dim %[[BIAS_CONST]], dims = [1] : (tensor<4xf32>) -> tensor<1x4x5x5xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x2x5x5xf32>) -> tensor<1x5x5x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[WEIGHT_CONST]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[BROADCAST_IN_DIM]], dims = [0, 2, 3, 1] : (tensor<1x4x5x5xf32>) -> tensor<1x5x5x4xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %[[CONV]], %[[TRANSPOSE_1]] : tensor<1x5x5x4xf32>
// CHECK: %[[MAX:.+]] = stablehlo.maximum %[[ADD]], %[[ZERO_CONST]] : tensor<1x5x5x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[MAX]], dims = [0, 3, 1, 2] : (tensor<1x5x5x4xf32>) -> tensor<1x4x5x5xf32>
// CHECK: return %[[TRANSPOSE_1]]
