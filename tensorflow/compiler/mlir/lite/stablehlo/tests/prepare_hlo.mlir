// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: odml-to-stablehlo-opt %s -prepare-hlo -split-input-file | FileCheck %s --dump-input=fail 

// Just assert that pass is properly registered.
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0: tensor<f32>
}
// CHECK-LABEL: main

// -----

//===----------------------------------------------------------------------===//
// stablehlo.convolution
//===----------------------------------------------------------------------===//

// 2D
//=--

// CHECK-LABEL: transpose_conv2d_same_padding_nchw_ihwo
func.func @transpose_conv2d_same_padding_nchw_ihwo(%input: tensor<1x2x256x256xf32>, %filter:tensor<2x2x4x4xf32>) -> tensor<1x2x512x512xf32> {
  %1 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1],
    window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x2x256x256xf32>, tensor<2x2x4x4xf32>) -> tensor<1x2x512x512xf32>
    func.return %1 : tensor<1x2x512x512xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.transpose %arg1
// CHECK-SAME: dims
// CHECK-SAME: [1, 2, 3, 0]
// CHECK:      %[[CONV_OUT:.*]] = stablehlo.convolution(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      stablehlo.transpose %[[CONV_OUT]]
// CHECK-SAME: dims
// CHECK-SAME: [0, 3, 1, 2]

// -----

// CHECK-LABEL: transpose_conv2d_same_padding_nchw_oihw
func.func @transpose_conv2d_same_padding_nchw_oihw(%input: tensor<1x2x256x256xf32>, %filter:tensor<2x2x4x4xf32>) -> tensor<1x2x512x512xf32> {
  %0 = stablehlo.convolution(%input, %filter)
   dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
   window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
    } : (tensor<1x2x256x256xf32>, tensor<2x2x4x4xf32>) -> tensor<1x2x512x512xf32>
  func.return %0 : tensor<1x2x512x512xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.transpose %arg1
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[CONV_OUT:.*]] = stablehlo.convolution(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      stablehlo.transpose %[[CONV_OUT]]
// CHECK-SAME: dims
// CHECK-SAME: [0, 3, 1, 2]

// -----

// CHECK-LABEL: depthwise_transpose_conv2d_same_padding_nchw_hwoi
func.func @depthwise_transpose_conv2d_same_padding_nchw_hwoi(%input: tensor<1x2x20x20xf32>, %filter:tensor<8x8x2x1xf32>) -> tensor<1x2x80x80xf32> {
  %1 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1],
    window = {pad = [[5, 5], [5, 5]], lhs_dilate = [4, 4]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64}
    : (tensor<1x2x20x20xf32>, tensor<8x8x2x1xf32>) -> tensor<1x2x80x80xf32>
  func.return %1 : tensor<1x2x80x80xf32>

  // CHECK:  %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<1x2x20x20xf32>) -> tensor<1x20x20x2xf32>
  // CHECK:  %1 = stablehlo.transpose %arg1, dims = [2, 0, 1, 3] : (tensor<8x8x2x1xf32>) -> tensor<2x8x8x1xf32>
  // CHECK:  %2 = stablehlo.slice %0 [0:1, 0:20, 0:20, 0:1]
  // CHECK:  %3 = stablehlo.slice %1 [0:1, 0:8, 0:8, 0:1]
  // CHECK:  %4 = stablehlo.convolution(%2, %3)
  // CHECK:  %5 = stablehlo.slice %0 [0:1, 0:20, 0:20, 1:2]
  // CHECK:  %6 = stablehlo.slice %1 [1:2, 0:8, 0:8, 0:1]
  // CHECK:  %7 = stablehlo.convolution(%5, %6)
  // CHECK:  %8 = stablehlo.concatenate %4, %7, dim = 3
  // CHECK:  %9 = stablehlo.transpose %8, dims = [0, 3, 1, 2]
  // CHECK:  return %9 : tensor<1x2x80x80xf32>
}

// -----

// CHECK-LABEL: grouped_transpose_conv2d_nhwc_ohwi
func.func @grouped_transpose_conv2d_nhwc_ohwi(%input: tensor<1x8x8x4xf32>, %filter: tensor<8x3x3x2xf32>) -> tensor<1x17x17x8xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64}
    : (tensor<1x8x8x4xf32>, tensor<8x3x3x2xf32>) -> tensor<1x17x17x8xf32>
  func.return %0 : tensor<1x17x17x8xf32>

  // CHECK:  %0 = stablehlo.slice %arg0 [0:1, 0:8, 0:8, 0:2] : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x2xf32>
  // CHECK:  %1 = stablehlo.slice %arg1 [0:4, 0:3, 0:3, 0:2] : (tensor<8x3x3x2xf32>) -> tensor<4x3x3x2xf32>
  // CHECK:  %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {pad = {{\[\[}}2, 2], [2, 2]], lhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x8x2xf32>, tensor<4x3x3x2xf32>) -> tensor<1x17x17x4xf32>
  // CHECK:  %3 = stablehlo.slice %arg0 [0:1, 0:8, 0:8, 2:4] : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x2xf32>
  // CHECK:  %4 = stablehlo.slice %arg1 [4:8, 0:3, 0:3, 0:2] : (tensor<8x3x3x2xf32>) -> tensor<4x3x3x2xf32>
  // CHECK:  %5 = stablehlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {pad = {{\[\[}}2, 2], [2, 2]], lhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x8x2xf32>, tensor<4x3x3x2xf32>) -> tensor<1x17x17x4xf32>
  // CHECK:  %6 = stablehlo.concatenate %2, %5, dim = 3 : (tensor<1x17x17x4xf32>, tensor<1x17x17x4xf32>) -> tensor<1x17x17x8xf32>
  // CHECK:  return %6 : tensor<1x17x17x8xf32>
}

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc
func.func @conv2d_nhwc_ohwi_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_dynamic
func.func @conv2d_nhwc_ohwi_nhwc_dynamic(%input: tensor<?x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<?x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc
func.func @conv2d_nchw_ohwi_nhwc(%input: tensor<?x3x256x256xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x3x256x256xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      stablehlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc_dynamic_batch
func.func @conv2d_nchw_ohwi_nhwc_dynamic_batch(%input: tensor<?x3x256x256xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x3x256x256xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      stablehlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nhwc_hwio_nhwc
func.func @conv2d_nhwc_hwio_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.transpose %arg1
// CHECK-SAME: dims
// CHECK-SAME: [3, 0, 1, 2]
// CHECK:      stablehlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nchw
func.func @conv2d_nhwc_ohwi_nchw(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x2x256x256xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x2x256x256xf32>
  func.return %0 : tensor<1x2x256x256xf32>
}

// CHECK-NOT:  transpose
// CHECK:      %[[CONV_OUT:.*]] = stablehlo.convolution
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      stablehlo.transpose %[[CONV_OUT]]
// CHECK-SAME: dims
// CHECK-SAME: [0, 3, 1, 2]

// -----

// CHECK-LABEL: conv2d_nchw_oihw_nchw
func.func @conv2d_nchw_oihw_nchw(%input: tensor<1x3x256x256xf32>, %filter: tensor<2x3x1x1xf32>) -> tensor<1x2x256x256xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x3x256x256xf32>, tensor<2x3x1x1xf32>) -> tensor<1x2x256x256xf32>
  func.return %0 : tensor<1x2x256x256xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// The kernel transpose only moves size-1 dims, so it is simplified to a reshape.
// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.reshape %arg1
// CHECK-SAME: (tensor<2x3x1x1xf32>) -> tensor<2x1x1x3xf32>
// CHECK:      %[[CONV_OUT:.*]] = stablehlo.convolution(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      stablehlo.transpose %[[CONV_OUT]]
// CHECK-SAME: dims
// CHECK-SAME: [0, 3, 1, 2]

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_padded
func.func @conv2d_nhwc_ohwi_nhwc_padded(%input: tensor<1x254x254x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<1> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x254x254x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad
// CHECK-SAME: low = [0, 1, 1, 0], high = [0, 1, 1, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]]
// CHECK-SAME: pad
// CHECK-SAME: [0, 0], [0, 0]
// CHECK-SAME: (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_asymmetric_padded
func.func @conv2d_nhwc_ohwi_nhwc_asymmetric_padded(%input: tensor<1x255x255x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x255x255x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad
// CHECK-SAME: low = [0, 0, 0, 0], high = [0, 1, 1, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]]
// CHECK-SAME: pad
// CHECK-SAME: [0, 0], [0, 0]
// CHECK-SAME: (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>


// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc_padded
func.func @conv2d_nchw_ohwi_nhwc_padded(%input: tensor<1x3x253x249xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[1, 2], [3, 4]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x3x253x249xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// Want to ensure that we transpose before padding input (which this test does implicitly).

// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad
// CHECK-SAME: low = [0, 1, 3, 0], high = [0, 2, 4, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]], %arg1)
// CHECK-SAME: pad
// CHECK-SAME: [0, 0], [0, 0]
// CHECK-SAME: (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>

// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc_padded_dilated_lhs
func.func @conv2d_nchw_ohwi_nhwc_padded_dilated_lhs(%input: tensor<1x64x64x256xf32>, %filter: tensor<64x2x2x256xf32>) -> tensor<1x128x128x64xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 2, 2>,
    padding = dense<1> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_reversal = array<i1: false, false>,
    window_strides = array<i64: 1, 1>} :
  (tensor<1x64x64x256xf32>, tensor<64x2x2x256xf32>) -> tensor<1x128x128x64xf32>
  func.return %0 : tensor<1x128x128x64xf32>
}

// CHECK-NOT:  stablehlo.pad
// CHECK:      stablehlo.convolution
// CHECK-SAME: pad
// CHECK-SAME: [1, 1], [1, 1]
// CHECK-SAME: lhs_dilate = [2, 2]

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ohwi_nhwc
func.func @depthwise_conv2d_nhwc_ohwi_nhwc(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<3312x3x3x1xf32>) -> tensor<1x8x8x3312xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x10x10x207xf32>, tensor<3312x3x3x1xf32>) -> tensor<1x8x8x3312xf32>
  func.return %0 : tensor<1x8x8x3312xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.transpose %arg1
// CHECK-SAME: dims
// CHECK-SAME: [3, 1, 2, 0]
// CHECK:      stablehlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----


// CHECK-LABEL: depthwise_conv2d_nchw_ihwo_nhwc
func.func @depthwise_conv2d_nchw_ihwo_nhwc(%arg0: tensor<1x207x10x10xf32>, %arg1: tensor<1x3x3x3312xf32>) -> tensor<1x8x8x3312xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x207x10x10xf32>, tensor<1x3x3x3312xf32>) -> tensor<1x8x8x3312xf32>
  func.return %0 : tensor<1x8x8x3312xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      stablehlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: depthwise_conv2d_nchw_ihwo_nhwc_padded
func.func @depthwise_conv2d_nchw_ihwo_nhwc_padded(%arg0: tensor<1x207x8x8xf32>, %arg1: tensor<1x3x3x3312xf32>) -> tensor<1x8x8x3312xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<1> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x207x8x8xf32>, tensor<1x3x3x3312xf32>) -> tensor<1x8x8x3312xf32>
  func.return %0 : tensor<1x8x8x3312xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad %[[TRANSPOSED_INPUT]]
// CHECK-SAME: low = [0, 1, 1, 0], high = [0, 1, 1, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]], %arg1
// CHECK-SAME: [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]
// CHECK-SAME: pad =
// CHECK-SAME: [0, 0], [0, 0]

// -----

// 3D
//=--

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc
func.func @conv3d_ndhwc_dhwio_ndhwc(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK-NOT:  transpose
// CHECK:      stablehlo.convolution
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ncdhw_dhwio_ndhwc
func.func @conv3d_ncdhw_dhwio_ndhwc(%arg0: tensor<1x207x8x8x32xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0, 1, 2]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x207x8x8x32xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = stablehlo.transpose %arg0
// CHECK-SAME: dims
// CHECK-SAME: [0, 2, 3, 4, 1]
// CHECK:      stablehlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ndhwc_odhwi_ndhwc
func.func @conv3d_ndhwc_odhwi_ndhwc(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<16x3x3x32x207xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[o, 0, 1, 2, i]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<16x3x3x32x207xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = stablehlo.transpose %arg1
// CHECK-SAME: dims
// CHECK-SAME: [1, 2, 3, 4, 0]
// CHECK:      stablehlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ncdhw
func.func @conv3d_ndhwc_dhwio_ncdhw(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x16x6x6x1xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, f, 0, 1, 2]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x16x6x6x1xf32>
  func.return %0 : tensor<1x16x6x6x1xf32>
}

// CHECK-NOT:  transpose
// CHECK:      %[[CONV_OUT:.*]] = stablehlo.convolution
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK:      stablehlo.transpose %[[CONV_OUT]]
// CHECK-SAME: dims
// CHECK-SAME: [0, 4, 1, 2, 3]

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_padded
func.func @conv3d_ndhwc_dhwio_ndhwc_padded(%arg0: tensor<1x6x6x30x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<3x2xi64>} :
       (tensor<1x6x6x30x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad
// CHECK-SAME: low = [0, 1, 1, 1, 0], high = [0, 1, 1, 1, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]], %arg1)
// CHECK-SAME: pad =
// CHECK-SAME: [0, 0], [0, 0], [0, 0]
// CHECK-SAME: (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>

// -----

// CHECK-LABEL: conv3d_ncdhw_dhwio_ndhwc_padded
func.func @conv3d_ncdhw_dhwio_ndhwc_padded(%arg0: tensor<1x207x6x6x30xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0, 1, 2]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<3x2xi64>} :
       (tensor<1x207x6x6x30xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[PADDED_LHS:.*]] = stablehlo.pad
// CHECK-SAME: low = [0, 1, 1, 1, 0], high = [0, 1, 1, 1, 0]
// CHECK:      stablehlo.convolution(%[[PADDED_LHS]], %arg1)
// CHECK-SAME: pad =
// CHECK-SAME: [0, 0], [0, 0], [0, 0]
// CHECK-SAME: (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>

// -----

// 1D
//=--

// CHECK-LABEL: conv1d_nsc_osi_nsc
func.func @conv1d_nsc_osi_nsc(%arg0: tensor<16x32x256xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1>,
    padding = dense<0> : tensor<1x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1>,
    window_strides = array<i64: 1>
  } : (tensor<16x32x256xf32>, tensor<256x1x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK: %[[RESHAPED_LHS:.*]] = "tfl.expand_dims"(%arg0
// CHECK: %[[RESHAPED_RHS:.*]] = "tfl.expand_dims"(%arg1
// CHECK: %[[CONV_OUT:.*]] = stablehlo.convolution(%[[RESHAPED_LHS]], %[[RESHAPED_RHS]]) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK: "tfl.squeeze"(%[[CONV_OUT]]

// -----

// CHECK-LABEL: conv1d_nsc_sio_nsc
func.func @conv1d_nsc_sio_nsc(%arg0: tensor<16x32x256xf32>, %arg1: tensor<1x256x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1>,
    padding = dense<0> : tensor<1x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1>,
    window_strides = array<i64: 1>
  } : (tensor<16x32x256xf32>, tensor<1x256x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK: %[[RESHAPED_LHS:.*]] = "tfl.expand_dims"(%arg0
// CHECK: %[[RESHAPED_RHS:.*]] = "tfl.expand_dims"(%arg1
// CHECK: %[[TPOSED_RHS:.*]] = stablehlo.transpose %[[RESHAPED_RHS]], dims = [3, 0, 1, 2] : (tensor<1x1x256x256xf32>) -> tensor<256x1x1x256xf32>
// CHECK: %[[CONV_OUT:.*]] = stablehlo.convolution(%[[RESHAPED_LHS]], %[[TPOSED_RHS]]) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK: "tfl.squeeze"(%[[CONV_OUT]]

// -----

// CHECK-LABEL: conv1d_ncs_osi_nsc_padded
func.func @conv1d_ncs_osi_nsc_padded(%arg0: tensor<16x256x30xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<1x2xi64>
  } : (tensor<16x256x30xf32>, tensor<256x1x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK: %[[RESHAPED_LHS:.*]] = "tfl.expand_dims"(%arg0{{.*}}-> tensor<16x256x30x1xf32>
// CHECK: %[[RESHAPED_RHS:.*]] = "tfl.expand_dims"(%arg1{{.*}}-> tensor<256x1x1x256xf32>
// CHECK: %[[TPOSED_LHS:.*]] = stablehlo.transpose %[[RESHAPED_LHS]], dims = [0, 2, 3, 1] : (tensor<16x256x30x1xf32>) -> tensor<16x30x1x256xf32>
// CHECK: %[[PADDED_LHS:.*]] = stablehlo.pad %[[TPOSED_LHS]], %cst, low = [0, 1, 0, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0] : (tensor<16x30x1x256xf32>, tensor<f32>) -> tensor<16x32x1x256xf32>
// CHECK: %[[CONV_OUT:.*]] = stablehlo.convolution(%[[PADDED_LHS]], %[[RESHAPED_RHS]]) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK: "tfl.squeeze"(%[[CONV_OUT]]{{.*}}-> tensor<16x32x256xf32>

// -----

// CHECK-LABEL: conv1d_ncs_osi_nsc_padded_dynamic
func.func @conv1d_ncs_osi_nsc_padded_dynamic(%arg0: tensor<?x256x30xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<?x32x256xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f, 0]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<1x2xi64>
  } : (tensor<?x256x30xf32>, tensor<256x1x256xf32>) -> tensor<?x32x256xf32>
  func.return %0 : tensor<?x32x256xf32>
}

// CHECK: %[[RESHAPED_LHS:.*]] = "tfl.expand_dims"(%arg0{{.*}}-> tensor<?x256x30x1xf32>
// CHECK: %[[RESHAPED_RHS:.*]] = "tfl.expand_dims"(%arg1{{.*}}-> tensor<256x1x1x256xf32>
// CHECK: %[[TPOSED_LHS:.*]] = stablehlo.transpose %[[RESHAPED_LHS]], dims = [0, 2, 3, 1] : (tensor<?x256x30x1xf32>) -> tensor<?x30x1x256xf32>
// CHECK: %[[PADDED_LHS:.*]] = stablehlo.pad %[[TPOSED_LHS]], %cst, low = [0, 1, 0, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0] : (tensor<?x30x1x256xf32>, tensor<f32>) -> tensor<?x32x1x256xf32>
// CHECK: %[[CONV_OUT:.*]] = stablehlo.convolution(%[[PADDED_LHS]], %[[RESHAPED_RHS]]) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK: "tfl.squeeze"(%[[CONV_OUT]]{{.*}}-> tensor<?x32x256xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.pad
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pad_2d
func.func @pad_2d(%arg0: tensor<3x3xf32>, %arg1: tensor<f32>) -> tensor<4x3xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, -1>,
    edge_padding_high = array<i64: 1, 1>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<3x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// CHECK:      stablehlo.slice
// CHECK-SAME: (tensor<3x3xf32>) -> tensor<3x2xf32>
// CHECK:      stablehlo.pad
// CHECK-SAME: (tensor<3x2xf32>, tensor<f32>) -> tensor<4x3xf32>

// -----

// CHECK-LABEL: pad_2d_negative
func.func @pad_2d_negative(%arg0: tensor<3x3xf32>, %arg1: tensor<f32>) -> tensor<1x2xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: -1, -1>,
    edge_padding_high = array<i64: -1, 0>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<3x3xf32>, tensor<f32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// CHECK:      stablehlo.slice
// CHECK-SAME: (tensor<3x3xf32>) -> tensor<1x2xf32>
// CHECK-NOT:  stablehlo.pad

// -----

// CHECK-LABEL: pad_3d_mixed
func.func @pad_3d_mixed(%arg0: tensor<3x3x3xf32>, %arg1: tensor<f32>) -> tensor<3x3x3xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: -1, 1, 0>,
    edge_padding_high = array<i64: 1, -1, 0>,
    interior_padding = array<i64: 0, 0, 0>
  } : (tensor<3x3x3xf32>, tensor<f32>) -> tensor<3x3x3xf32>
  func.return %0 : tensor<3x3x3xf32>
}

// CHECK:      stablehlo.slice
// CHECK-SAME: (tensor<3x3x3xf32>) -> tensor<2x2x3xf32>
// CHECK:      stablehlo.pad
// CHECK-SAME: (tensor<2x2x3xf32>, tensor<f32>) -> tensor<3x3x3xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.reduce_window
//===----------------------------------------------------------------------===//

// CHECK-LABEL: reduce_window_valid_channel_first
func.func @reduce_window_valid_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x7x7xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 1, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2>} : (tensor<4x3x16x16xf32>, tensor<f32>) -> tensor<4x3x7x7xf32>
  func.return %1 : tensor<4x3x7x7xf32>
}

// CHECK:      %[[INIT_CST:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:      %[[TPOSE_IN:.*]] = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[RW:.*]] = "stablehlo.reduce_window"(%[[TPOSE_IN]], %[[INIT_CST]])
// CHECK-SAME: window_dimensions = array<i64: 1, 3, 3, 1>
// CHECK-SAME: window_strides = array<i64: 1, 2, 2, 1>
// CHECK:      %[[TPOSE_OUT:.*]] = stablehlo.transpose %[[RW]], dims = [0, 3, 1, 2] : (tensor<4x7x7x3xf32>) -> tensor<4x3x7x7xf32>

// -----

// CHECK-LABEL: reduce_window_same_channel_first
func.func @reduce_window_same_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x8x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 1], [0, 1]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 1, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2>} : (tensor<4x3x16x16xf32>, tensor<f32>) -> tensor<4x3x8x8xf32>
  func.return %1 : tensor<4x3x8x8xf32>
}

// CHECK:      %[[INIT_CST:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:      %[[TPOSE_IN:.*]] = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[RW:.*]] = "stablehlo.reduce_window"(%[[TPOSE_IN]], %[[INIT_CST]])
// CHECK-SAME: padding
// CHECK-SAME: [0, 0], [0, 1], [0, 1], [0, 0]
// CHECK-SAME: window_dimensions = array<i64: 1, 3, 3, 1>
// CHECK-SAME: window_strides = array<i64: 1, 2, 2, 1>
// CHECK:      %[[TPOSE_OUT:.*]] = stablehlo.transpose %[[RW]], dims = [0, 3, 1, 2] : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.dynamic_slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dynamic_slice
func.func @dynamic_slice(%arg0: tensor<7x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK:      stablehlo.dynamic_slice
// CHECK-SAME: (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_ui32
func.func @dynamic_slice_ui32(%arg0: tensor<7x3xf32>, %arg1: tensor<ui32>, %arg2: tensor<ui32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<ui32>, tensor<ui32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK:      stablehlo.dynamic_slice
// CHECK-SAME: (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>

// CHECK-LABEL: dynamic_slice_ui64
func.func @dynamic_slice_ui64(%arg0: tensor<7x3xf32>, %arg1: tensor<ui64>, %arg2: tensor<ui64>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<ui64>, tensor<ui64>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK:      stablehlo.dynamic_slice
// CHECK-SAME: (tensor<7x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_i64
func.func @dynamic_slice_i64(%arg0: tensor<7x3xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK:      stablehlo.dynamic_slice
// CHECK-SAME: (tensor<7x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x2xf32>

//===----------------------------------------------------------------------===//
// stablehlo.custom_call
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @shape_assertion_custom_call
func.func @shape_assertion_custom_call(%arg1: tensor<?x5xi32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<3> : tensor<i32>
  %1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x5xi32>) -> tensor<i32>
  %ok = stablehlo.compare  EQ, %1, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {
    error_message = "The error message",
    has_side_effect = true
  } : (tensor<i1>) -> ()
  return %1 : tensor<i32>
}

// CHECK-NOT: stablehlo.custom_call

//===----------------------------------------------------------------------===//
// stablehlo.fft
//===----------------------------------------------------------------------===//

// CHECK-LABEL: rfft_2d
func.func @rfft_2d(%arg0: tensor<1x512xf32>) -> tensor<1x257xcomplex<f32>> {
  %0 = stablehlo.fft %arg0, type = RFFT, length = [512] : (tensor<1x512xf32>) -> tensor<1x257xcomplex<f32>>
  func.return %0 : tensor<1x257xcomplex<f32>>
}

// CHECK:  %0 = stablehlo.fft %arg0, type = RFFT, length = [1, 512] : (tensor<1x512xf32>) -> tensor<1x257xcomplex<f32>>
// CHECK:  return %0 : tensor<1x257xcomplex<f32>>

// -----

// CHECK-LABEL: @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) <{ fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type FFT> }> : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// CHECK: %0 = stablehlo.fft %arg0, type = FFT, length = [9] : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
// CHECK: return %0 : tensor<3x9xcomplex<f32>>

// -----

// CHECK-LABEL: @mhlo_nd_fft
func.func @mhlo_nd_fft(%arg0: tensor<2x3x345x256xf32>) -> tensor<2x3x345x129xcomplex<f32>> {
  %43 = stablehlo.fft %arg0, type = RFFT, length = [256] : (tensor<2x3x345x256xf32>) -> tensor<2x3x345x129xcomplex<f32>>
  return %43 : tensor<2x3x345x129xcomplex<f32>>
}

// CHECK: %0 = stablehlo.reshape %arg0 : (tensor<2x3x345x256xf32>) -> tensor<2x3x345x1x256xf32>
// CHECK: %1 = stablehlo.fft %0, type = RFFT, length = [1, 256] : (tensor<2x3x345x1x256xf32>) -> tensor<2x3x345x1x129xcomplex<f32>>
// CHECK: %2 = stablehlo.reshape %1 : (tensor<2x3x345x1x129xcomplex<f32>>) -> tensor<2x3x345x129xcomplex<f32>>
// CHECK: return %2 : tensor<2x3x345x129xcomplex<f32>>

// -----

// CHECK-LABEL: @mhlo_dynamic_fft_1
func.func @mhlo_dynamic_fft_1(%arg0: tensor<?x9x2560xf32>) -> tensor<?x9x1281xcomplex<f32>> {
  %0 = stablehlo.fft %arg0, type = RFFT, length = [2560] : (tensor<?x9x2560xf32>) -> tensor<?x9x1281xcomplex<f32>>
  return %0 : tensor<?x9x1281xcomplex<f32>>
  // CHECK: stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: stablehlo.concatenate
  // CHECK: %[[EXPANDED:.*]] = stablehlo.dynamic_reshape %arg0
  // CHECK: %[[FFT_OUT:.*]] = stablehlo.fft %[[EXPANDED]], type = RFFT, length = [1, 2560]
  // CHECK: %[[COLLAPSED:.*]] = stablehlo.dynamic_reshape %[[FFT_OUT]]
  // CHECK: return %[[COLLAPSED]]
}

// -----

// CHECK-LABEL: @mhlo_dynamic_fft_2
func.func @mhlo_dynamic_fft_2(%arg0: tensor<?x?x2560xf32>) -> tensor<?x?x1281xcomplex<f32>> {
  %0 = stablehlo.fft %arg0, type = RFFT, length = [2560] : (tensor<?x?x2560xf32>) -> tensor<?x?x1281xcomplex<f32>>
  return %0 : tensor<?x?x1281xcomplex<f32>>
  // CHECK: stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: stablehlo.concatenate
  // CHECK: %[[EXPANDED:.*]] = stablehlo.dynamic_reshape %arg0
  // CHECK: %[[FFT_OUT:.*]] = stablehlo.fft %[[EXPANDED]], type = RFFT, length = [1, 2560]
  // CHECK: %[[COLLAPSED:.*]] = stablehlo.dynamic_reshape %[[FFT_OUT]]
  // CHECK: return %[[COLLAPSED]]
}

// -----

// CHECK-LABEL: @mhlo_dynamic_fft_2_neg
func.func @mhlo_dynamic_fft_2_neg(%arg0: tensor<?x9x?xf32>) -> tensor<?x9x1281xcomplex<f32>> {
  %0 = stablehlo.fft %arg0, type = RFFT, length = [2560] : (tensor<?x9x?xf32>) -> tensor<?x9x1281xcomplex<f32>>
  return %0 : tensor<?x9x1281xcomplex<f32>>
  // CHECK: %0 = stablehlo.fft %arg0, type = RFFT, length = [2560]
  // CHECK: return %0
}

// -----

// Transposes of non-splat constants are not folded (MHLO didn't either; the
// StableHLO folder does it element by element, which is slow on big weights).

// CHECK-LABEL: @transpose_non_splat_constant_not_folded
func.func @transpose_non_splat_constant_not_folded() -> tensor<3x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
  // CHECK: %[[T:.*]] = stablehlo.transpose %[[CST]], dims = [1, 0]
  // CHECK: return %[[T]]
}

// -----

// CHECK-LABEL: @transpose_splat_constant_folded
func.func @transpose_splat_constant_folded() -> tensor<3x2xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<2x3xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<3x2xf32>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
}
