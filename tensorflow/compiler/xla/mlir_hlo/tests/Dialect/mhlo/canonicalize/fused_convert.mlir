// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: @dot_general_fuse_both_with_attrs
func.func @dot_general_fuse_both_with_attrs(%arg0: tensor<16x64x128xf16>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = mhlo.convert(%arg0) : (tensor<16x64x128xf16>) -> tensor<16x64x128xf32>
  %1 = mhlo.convert(%arg1) : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: "mhlo.dot_general"(%arg0, %arg1)
    // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    // CHECK-SAME: -> tensor<16x64x3072xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_general_fuse_one
func.func @dot_general_fuse_one(%arg0: tensor<16x64x128xf64>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = mhlo.convert(%arg0) : (tensor<16x64x128xf64>) -> tensor<16x64x128xf32>
  %1 = mhlo.convert(%arg1) : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: %[[CONVERT:.+]] = mhlo.convert(%arg0)
  // CHECK: "mhlo.dot_general"(%[[CONVERT]], %arg1)
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_basic
func.func @dot_basic(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = mhlo.convert(%arg0) : (tensor<4x4xf16>) -> tensor<4x4xf32>
  %1 = mhlo.convert(%arg1) : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1)
  %2 = "mhlo.dot"(%0, %1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[DOT]]
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @convolution
func.func @convolution(%arg0: tensor<16x32x256xbf16>, %arg1: tensor<1x256x256xbf16>) -> tensor<16x32x256xf32> {
  %cast = mhlo.convert(%arg0) : (tensor<16x32x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: %[[CONV:.+]] = mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: -> tensor<16x32x256xf32>
  %0 = "mhlo.convolution"(%cast, %arg1) {
     batch_group_count = 1 : i64,
     dimension_numbers = #mhlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
     feature_group_count = 1 : i64,
     lhs_dilation = dense<1> : tensor<1xi64>,
     padding = dense<0> : tensor<1x2xi64>,
     precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
     rhs_dilation = dense<1> : tensor<1xi64>,
     window_strides = dense<1> : tensor<1xi64>
   } : (tensor<16x32x256xf32>, tensor<1x256x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: return %[[CONV]]
  func.return %0 : tensor<16x32x256xf32>
}

