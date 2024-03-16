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
