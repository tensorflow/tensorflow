// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Valid: Generic convolution

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32> {
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

// Valid: Test convolution i8xi8 -> i32.

// CHECK-LABEL: func @convolution_upcast
func.func @convolution_upcast(%arg0 : tensor<100x26x26x32xi8>,
    %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

// -----

// Valid: Empty spatial dimensions

// CHECK: func @conv_empty_spatial_dimensions
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, f]x[i, o]->[b, f]
// CHECK-SAME: window = {stride = [], pad = [], lhs_dilate = [],
// CHECK-SAME: rhs_dilate = [], reverse = []}
func.func @conv_empty_spatial_dimensions(%arg0: tensor<3x2xf16>,
    %arg1: tensor<2x2xf16>) -> tuple<tensor<3x2xf16>> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, f]x[i, o]->[b, f],
         window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [],
           reverse = []}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         }
       : (tensor<3x2xf16>, tensor<2x2xf16>) -> tensor<3x2xf16>
  %1 = "mhlo.tuple"(%0) : (tensor<3x2xf16>) -> tuple<tensor<3x2xf16>>
  func.return %1 : tuple<tensor<3x2xf16>>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<2x4x5x2xf32>,
                     %arg1: tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32> {
  // expected-error@+1 {{expects input dimension-numbers to be unique, got {0, 0}.}}
  %1 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      output_batch_dimension = 0,
      output_feature_dimension = 3,
    >,
    feature_group_count = 2 : i64,
    someattr} : (tensor<2x4x5x2xf32>, tensor<2x2x1x6xf32>) ->
      tensor<2x3x4x6xf32>
  func.return %1 : tensor<2x3x4x6xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects convolution arguments to have same number of dimensions. Got: 'tensor<1x8x8x207xf32>' and 'tensor<3x3x207xf32>'.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1xf32>, %arg1: tensor<3xf32>)
    -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects convolution arguments to have >= 2 dimensions. Got: 'tensor<1xf32>' and 'tensor<3xf32>'.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1xf32>, tensor<3xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects the same size for input, kernel and output spatial-dimensions, but got 3, 2, and 2 resp.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, 2, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects the same size for input, kernel and output spatial-dimensions, but got 2, 3, and 2 resp.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, 2, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects the same size for input, kernel and output spatial-dimensions, but got 2, 2, and 3 resp.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, 2, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+1 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 4],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+1 {{expects kernel dimension-numbers to be unique, got {3, 2, 0, 0}.}}
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 0],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+1 {{expects output dimension-numbers to be unique, got {0, 3, 0, 3}.}}
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [0, 3]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+5 {{op attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 0 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+5 {{op attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 0 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects batch_group_count and feature_group_count not to be both greater than 1. Got 2 and 2 resp.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 2 : i64,
           feature_group_count = 2 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<3x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<3x8x8x16xf32> {
  // expected-error@+1 {{expects output feature dimension size (16) to be a multiple of batch_group_count. Got batch_group_count = 3.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 3 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<3x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<3x8x8x16xf32>
  func.return %0 : tensor<3x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects input feature dimension (207) to be a multiple of feature_group_count. Got feature_group_count = 2.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 2 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects input feature dimension (207) / feature_group_count = kernel input feature dimension (20). Got feature_group_count = 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x69x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects kernel output feature dimension (16) to be divisible by feature_group_count. For feature_group_count = 3.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 3 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x69x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<5x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects input batch dimension (5) to be divisible by batch_group_count. Got batch_group_count = 2.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 2 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<5x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window-strides to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects base-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window-reversal to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects padding-entries to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         // expected-error@+1 {{Expected array with 2 elements, got 4 elements instead}}
         window = {stride = [1, 1], pad = [[1, 1, 1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_dimensions(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32> {
  // expected-error@+1 {{expects padding-entries to have same dimension-size as size of window dimensions (2), but got: 3.}}
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<0x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window to have positive value for 0-th window dimension, but got 0.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<0x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window to have positive stride for 1-th window dimension, but got 0.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 0], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window to have positive base dilation factor for 0-th window dimension, but got 0.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [0, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

func.func @invalid_conv_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+1 {{expects window to have positive window dilation factor for 0-th window dimension, but got 0.}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [0, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

// Invalid rank of output-type.

func.func @invalid_conv_return_type(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x16xf32> {
  // expected-error @+1 {{expects rank of convolution return-type to be equal to input-ranks (4), but got 3}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x16xf32>
  func.return %0 : tensor<1x8x16xf32>
}

// -----

// Invalid batch dimension in output-type. Should be equal to
// input-batch-dimension / batch_group_count.

func.func @invalid_conv_return_type(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<2x8x8x16xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, 8, 16]' is incompatible with return type of operation 'tensor<2x8x8x16xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<2x8x8x16xf32>
  func.return %0 : tensor<2x8x8x16xf32>
}

// -----

// Invalid feature dimension in output-type. Should be equal to
// kernel_output_feature_dimension.

func.func @invalid_conv_return_type(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x32xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, 8, 16]' is incompatible with return type of operation 'tensor<1x8x8x32xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x32xf32>
  func.return %0 : tensor<1x8x8x32xf32>
}

// -----

// The following tests checks the inferred output-type of ConvolutionOp. We
// deliberately put an invalid output-type in these tests so that the
// inffered-type can be highlighted in the error message.

// Dynamic input-batch-dimension
func.func @invalid_conv_dynamic_shapes(%arg0: tensor<?x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[?, 8, 8, 16]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<?x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

// Dynamic input-feature-dimension: No effect on output dimensions.
func.func @invalid_conv_dynamic_shapes(%arg0: tensor<1x8x8x?xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, 8, 16]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x?xf32>, tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

// Dynamic input-spatial-dimension
func.func @invalid_conv_dynamic_shapes(%arg0: tensor<1x?x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[1, ?, 8, 16]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x?x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

// Dynamic kernel-input-feature-dimension: No effect on output dimensions.
func.func @invalid_conv_dynamic_shapes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x?x16xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, 8, 16]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x?x16xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

// Dynamic kernel-output-feature-dimension
func.func @check_inferred_type_with_dynamic_input_dims(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x?xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, 8, ?]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x?xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

// Dynamic kernel-spatial-dimension
func.func @check_inferred_type_with_dynamic_input_dims(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x?x207x16xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error@+1 {{inferred shape '[1, 8, ?, 16]' is incompatible with return type of operation 'tensor<1x1x1x1xf32>'}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1,1]],
           lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         } :
       (tensor<1x8x8x207xf32>, tensor<3x?x207x16xf32>) -> tensor<1x1x1x1xf32>
  func.return %0 : tensor<1x1x1x1xf32>
}

// -----

func.func @conv2d(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error @+3 {{'mhlo.convolution' Expected array with 2 elements, got 3 elements instead}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1, 1], [1, 1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

// CHECK: module
// CHECK-SAME: mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 1, 0, f]>
module attributes { mhlo.conv = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [2, 1]>} {}

// -----

// CHECK: module
// CHECK: mhlo.conv = #mhlo.conv<[b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f]>
module attributes {
  mhlo.conv = #mhlo.conv<[b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension c, expecting b, f}}
  mhlo.conv = #mhlo.conv<[c, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension b, expecting i, o}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, b, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension i, expecting o}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, i]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Expected dimensions f not specified}}
  mhlo.conv = #mhlo.conv<[b, 0, 1]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected keyword b}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o, b]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{expected '['}}
  mhlo.conv = #mhlo.conv<{b, 0, 1, f}x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Expected spatial dimensions 0 not specified}}
  mhlo.conv = #mhlo.conv<[b, f, 1]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

module attributes {
  // expected-error@+1{{Duplicate entries for spatial dimension 1}}
  mhlo.conv = #mhlo.conv<[b, f, 1, 0, 1]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension -2}}
  mhlo.conv = #mhlo.conv<[b, f, 1, -2]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{Unexpected keyword stide}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stide = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}

// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{expected integer value}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, b], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}

// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{Unexpected keyword stride}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2], stride=[2,1]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}

// -----

func.func @conv_invalid_precision_config(%arg0: tensor<3x2xf16>,
    %arg1: tensor<2x2xf16>) -> tuple<tensor<3x2xf16>> {
  // expected-error@+1 {{expects precision config to be empty or have <= 2 elements}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, f]x[i, o]->[b, f],
         window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [],
           reverse = []}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
         }
       : (tensor<3x2xf16>, tensor<2x2xf16>) -> tensor<3x2xf16>
  %1 = "mhlo.tuple"(%0) : (tensor<3x2xf16>) -> tuple<tensor<3x2xf16>>
  func.return %1 : tuple<tensor<3x2xf16>>
}
