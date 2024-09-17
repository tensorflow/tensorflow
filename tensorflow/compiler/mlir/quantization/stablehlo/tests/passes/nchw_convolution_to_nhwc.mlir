// RUN: stablehlo-quant-opt %s -stablehlo-nchw-convolution-to-nhwc \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// Tests that `stablehlo.transpose` ops are inserted for each of input, filter,
// and output.
// Output dimension numbers =  [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]

// CHECK-LABEL: nchw_conv
// CHECK-SAME: %[[ARG:.+]]: tensor<1x8x4x4xf32>
func.func @nchw_conv(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %0 = stablehlo.constant() {value = dense<7.000000e+00> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x4x4xf32>, tensor<8x8x3x3xf32>) -> tensor<1x8x4x4xf32>
  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-DAG: %[[CONST:.+]] = stablehlo.constant {{.*}} : tensor<8x8x3x3xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 2, 3, 1] : (tensor<1x8x4x4xf32>) -> tensor<1x4x4x8xf32>
// CHECK-DAG: %[[TRANSPOSE_1:.+]] = stablehlo.transpose %[[CONST]], dims = [2, 3, 1, 0] : (tensor<8x8x3x3xf32>) -> tensor<3x3x8x8xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[TRANSPOSE_1]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = {{\[\[}}1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x4x8xf32>, tensor<3x3x8x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK: %[[TRANSPOSE_2:.+]] = stablehlo.transpose %[[CONV]], dims = [0, 3, 1, 2] : (tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32>

// -----

// Tests that the conversion doesn't happen when the input dimension numbers
// are not [b, f, 0, 1].

// CHECK-LABEL: conv_input_dim_numbers_mismatch
func.func @conv_input_dim_numbers_mismatch(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {
  %0 = stablehlo.constant() {value = dense<7.000000e+00> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x4x8xf32>, tensor<8x8x3x3xf32>) -> tensor<1x8x4x4xf32>
  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-NOT: stablehlo.transpose
// CHECK: %[[CONV:.+]] = stablehlo.convolution
// CHECK-SAME{LITERAL}: [b, 0, 1, f]x[o, i, 0, 1]->[b, f, 0, 1]
// CHECK-NOT: stablehlo.transpose

// -----

// Tests that the conversion doesn't happen when the feature dimension numbers
// are not [i, 0, 1, o].

// CHECK-LABEL: conv_feature_dim_numbers_mismatch
func.func @conv_feature_dim_numbers_mismatch(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %0 = stablehlo.constant() {value = dense<7.000000e+00> : tensor<8x3x3x8xf32>} : () -> tensor<8x3x3x8xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[i, 0, 1, o]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x4x4xf32>, tensor<8x3x3x8xf32>) -> tensor<1x8x4x4xf32>
  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-NOT: stablehlo.transpose
// CHECK: %[[CONV:.+]] = stablehlo.convolution
// CHECK-SAME{LITERAL}: [b, f, 0, 1]x[i, 0, 1, o]->[b, f, 0, 1]
// CHECK-NOT: stablehlo.transpose

// -----

// Tests that the conversion doesn't happen when the output dimension numbers
// are not [b, 0, 1, f].

// CHECK-LABEL: conv_output_dim_numbers_mismatch
func.func @conv_output_dim_numbers_mismatch(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x4x4x8xf32> {
  %0 = stablehlo.constant() {value = dense<7.000000e+00> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x4x4xf32>, tensor<8x8x3x3xf32>) -> tensor<1x4x4x8xf32>
  return %2 : tensor<1x4x4x8xf32>
}

// CHECK-NOT: stablehlo.transpose
// CHECK: %[[CONV:.+]] = stablehlo.convolution
// CHECK-SAME{LITERAL}: [b, f, 0, 1]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-NOT: stablehlo.transpose

// -----

// Tests that a quantized convolution does not match. No conversion occurs.

// CHECK-LABEL: quantized_convolution
func.func @quantized_convolution(%arg0: tensor<1x4x3x3x!quant.uniform<i8:f32, 1.000000e+0:-100>>, %arg1: tensor<2x4x3x3x!quant.uniform<i8:f32:0, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x2x3x3x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x3x3x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<2x4x3x3x!quant.uniform<i8:f32:0, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x2x3x3x!quant.uniform<i8:f32, 4.000000e+0>>
  return %0 : tensor<1x2x3x3x!quant.uniform<i8:f32, 4.000000e+0>>
}

// CHECK-NOT: stablehlo.transpose

// -----

// Tests that a quantized convolution with rank > 4 does not match.
// No conversion occurs.

// CHECK-LABEL: convolution_3d
func.func @convolution_3d(%arg0: tensor<1x4x28x28x1xf32>, %arg1: tensor<2x3x3x1x16xf32>) -> tensor<1x3x26x26x16xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x28x28x1xf32>, tensor<2x3x3x1x16xf32>) -> tensor<1x3x26x26x16xf32>
  return %0 : tensor<1x3x26x26x16xf32>
}

// CHECK-NOT: stablehlo.transpose
