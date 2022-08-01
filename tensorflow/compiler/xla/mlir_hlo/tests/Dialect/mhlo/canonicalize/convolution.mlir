// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: @convolution_simple
func.func @convolution_simple(%arg0: tensor<5x6xf32>, %arg1: tensor<?x6xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%arg0, %arg1)
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[o, i]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<?x6xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @convolution_swap
func.func @convolution_swap(%arg0: tensor<5x6xf32>, %arg1: tensor<?x6xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%arg0, %arg1)
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = mhlo.convolution(%arg1, %arg0) dim_numbers = [b, f]x[o, i]->[f, b], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x6xf32>, tensor<5x6xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @dot_general_is_dot
func.func @dot_general_is_dot(%arg0: tensor<5x6xf32>, %arg1: tensor<6x?xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1) 
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<6x?xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @convolution_is_mul_simple
func.func @convolution_is_mul_simple(%arg0: tensor<5x6xf32>, %arg1: tensor<1x6xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[RESHAPE:.+]] = "mhlo.reshape"(%arg1) : (tensor<1x6xf32>) -> tensor<6xf32>
  // CHECK: %[[BROAD:.+]] = "mhlo.broadcast_in_dim"(%[[RESHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<6xf32>) -> tensor<5x6xf32>
  // CHECK: %[[MULTIPLY:.+]] = mhlo.multiply %arg0, %[[BROAD]]
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<1x6xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[MULTIPLY]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: @convolution_is_mul_rhs_transpose
func.func @convolution_is_mul_rhs_transpose(%arg0: tensor<5x6xf32>, %arg1: tensor<6x1xf32>) -> tensor<5x6xf32> {
  // CHECK: %[[RESHAPE:.+]] = "mhlo.reshape"(%arg1) : (tensor<6x1xf32>) -> tensor<6xf32>
  // CHECK: %[[BROAD:.+]] = "mhlo.broadcast_in_dim"(%[[RESHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<6xf32>) -> tensor<5x6xf32>
  // CHECK: %[[MULTIPLY:.+]] = mhlo.multiply %arg0, %[[BROAD]]
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[o, i]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf32>, tensor<6x1xf32>) -> tensor<5x6xf32>
  // CHECK: return %[[MULTIPLY]]
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: @convolution_is_mul_transpose_result
func.func @convolution_is_mul_transpose_result(%arg0: tensor<5x6xf16>, %arg1: tensor<6x1xf16>) -> tensor<6x5xf32> {
  // CHECK: %[[RESHAPE:.+]] = "mhlo.reshape"(%arg1) : (tensor<6x1xf16>) -> tensor<6xf16>
  // CHECK: %[[BROAD:.+]] = "mhlo.broadcast_in_dim"(%[[RESHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<6xf16>) -> tensor<5x6xf16>
  // CHECK: %[[CONVL:.+]] = mhlo.convert(%arg0) : (tensor<5x6xf16>) -> tensor<5x6xf32>
  // CHECK: %[[CONVR:.+]] = mhlo.convert(%[[BROAD]]) : (tensor<5x6xf16>) -> tensor<5x6xf32>
  // CHECK: %[[MULTIPLY:.+]] = mhlo.multiply %[[CONVL]], %[[CONVR]]
  // CHECK: %[[TRANSPOSE:.+]] = "mhlo.transpose"(%[[MULTIPLY]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[o, i]->[f, b], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 6 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x6xf16>, tensor<6x1xf16>) -> tensor<6x5xf32>
  // CHECK: %[[TRANSPOSE]]
  return %0 : tensor<6x5xf32>
}