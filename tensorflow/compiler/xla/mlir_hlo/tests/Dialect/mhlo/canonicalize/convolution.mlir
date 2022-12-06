// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: @dot_general_is_dot
func.func @dot_general_is_dot(%arg0: tensor<5x6xf32>, %arg1: tensor<6x?xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1) 
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<6x?xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @convolution_is_dot_general
func.func @convolution_is_dot_general(%arg0: tensor<5x6xf32>, %arg1: tensor<?x6xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%arg0, %arg1)
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[o, i]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<?x6xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @convolution_is_dot_general_swap
func.func @convolution_is_dot_general_swap(%arg0: tensor<5x6xf32>, %arg1: tensor<?x6xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%arg0, %arg1)
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = mhlo.convolution(%arg1, %arg0) dim_numbers = [b, f]x[o, i]->[f, b], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x6xf32>, tensor<5x6xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @conv_grouped_is_dot
func.func @conv_grouped_is_dot(%arg0: tensor<5x12xf32>, %arg1: tensor<2x6xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[RES0:.+]] = mhlo.reshape %arg0 : (tensor<5x12xf32>) -> tensor<5x6x2xf32>
  // CHECK: %[[RES1:.+]] = mhlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<6x1x2xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[RES0]], %[[RES1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%2) {permutation = dense<[1, 0, 2]> : tensor<3xi64>}
  // CHECK: %[[OUT:.+]] = mhlo.reshape %3 : (tensor<5x6x1xf32>) -> tensor<5x6xf32>
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x12xf32>, tensor<2x6xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[OUT]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: conv_grouped_is_dot_multi
func.func @conv_grouped_is_dot_multi(%arg0: tensor<5x4xf32>, %arg1: tensor<2x6xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[LHS:.+]] = mhlo.reshape %arg0 : (tensor<5x4xf32>) -> tensor<5x2x2xf32>
  // CHECK: %[[RHS:.+]] = mhlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x3x2xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[DOT]]) {permutation = dense<[1, 0, 2]> : tensor<3xi64>}
  // CHECK: %[[OUT:.+]] = mhlo.reshape %[[TRANSPOSE]] : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x4xf32>, tensor<2x6xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[OUT]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: conv_grouped_is_dot_transpose_rhs
func.func @conv_grouped_is_dot_transpose_rhs(%arg0: tensor<5x4xf32>, %arg1: tensor<6x2xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[LHS:.+]] = mhlo.reshape %arg0 : (tensor<5x4xf32>) -> tensor<5x2x2xf32>
  // CHECK: %[[RHS:.+]] = mhlo.reshape %arg1 : (tensor<6x2xf32>) -> tensor<2x2x3xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[DOT]]) {permutation = dense<[1, 0, 2]> : tensor<3xi64>}
  // CHECK: %[[OUT:.+]] = mhlo.reshape %[[TRANSPOSE]] : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[o, i]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x4xf32>, tensor<6x2xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[OUT]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: conv_grouped_is_dot_transpose_ins
func.func @conv_grouped_is_dot_transpose_ins(%arg0: tensor<4x5xf32>, %arg1: tensor<6x2xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[LHS:.+]] = mhlo.reshape %arg0 : (tensor<4x5xf32>) -> tensor<2x2x5xf32>
  // CHECK: %[[RHS:.+]] = mhlo.reshape %arg1 : (tensor<6x2xf32>) -> tensor<2x2x3xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[DOT]]) {permutation = dense<[1, 0, 2]> : tensor<3xi64>}
  // CHECK: %[[OUT:.+]] = mhlo.reshape %[[TRANSPOSE]] : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b]x[o, i]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x5xf32>, tensor<6x2xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[OUT]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: conv_grouped_is_dot_transpose_out
func.func @conv_grouped_is_dot_transpose_out(%arg0: tensor<5x4xf32>, %arg1: tensor<2x6xf32>) -> tensor<6x5xf32> {
  // CHECK: %[[LHS:.+]] = mhlo.reshape %arg0 : (tensor<5x4xf32>) -> tensor<5x2x2xf32>
  // CHECK: %[[RHS:.+]] = mhlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x3x2xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[DOT]]) {permutation = dense<[0, 2, 1]> : tensor<3xi64>}
  // CHECK: %[[OUT:.+]] = mhlo.reshape %[[TRANSPOSE]]
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[f, b], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x4xf32>, tensor<2x6xf32>) -> tensor<6x5xf32>
  // CHECK: return %[[OUT]]
  return %0 : tensor<6x5xf32>
}