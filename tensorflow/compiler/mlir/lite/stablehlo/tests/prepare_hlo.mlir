// RUN: odml-to-stablehlo-opt %s -prepare-hlo -split-input-file | FileCheck %s --dump-input=fail 

// Just assert that pass is properly registered.
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0: tensor<f32>
}
// CHECK-LABEL: main

// -----

//===----------------------------------------------------------------------===//
// mhlo.convolution
//===----------------------------------------------------------------------===//

// 2D
//=--

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc
func.func @conv2d_nhwc_ohwi_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc
func.func @conv2d_nchw_ohwi_nhwc(%input: tensor<?x3x256x256xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x3x256x256xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      mhlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nchw_ohwi_nhwc_dynamic_batch
func.func @conv2d_nchw_ohwi_nhwc_dynamic_batch(%input: tensor<?x3x256x256xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x3x256x256xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      mhlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nhwc_hwio_nhwc
func.func @conv2d_nhwc_hwio_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = "mhlo.transpose"(%arg1)
// CHECK-SAME: permutation
// CHECK-SAME: [3, 0, 1, 2]
// CHECK:      mhlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nchw
func.func @conv2d_nhwc_ohwi_nchw(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x2x256x256xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x2x256x256xf32>
  func.return %0 : tensor<1x2x256x256xf32>
}

// CHECK-NOT:  transpose
// CHECK:      %[[CONV_OUT:.*]] = mhlo.convolution
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      "mhlo.transpose"(%[[CONV_OUT]])
// CHECK-SAME: permutation
// CHECK-SAME: [0, 3, 1, 2]

// -----

// CHECK-LABEL: conv2d_nchw_oihw_nchw
func.func @conv2d_nchw_oihw_nchw(%input: tensor<1x3x256x256xf32>, %filter: tensor<2x3x1x1xf32>) -> tensor<1x2x256x256xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x3x256x256xf32>, tensor<2x3x1x1xf32>) -> tensor<1x2x256x256xf32>
  func.return %0 : tensor<1x2x256x256xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = "mhlo.transpose"(%arg1)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 3, 1]
// CHECK:      %[[CONV_OUT:.*]] = mhlo.convolution(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK:      "mhlo.transpose"(%[[CONV_OUT]])
// CHECK-SAME: permutation
// CHECK-SAME: [0, 3, 1, 2]

// -----

// 1D
//=--

// CHECK-LABEL: conv1d_nsc_osi_nsc
func.func @conv1d_nsc_osi_nsc(%arg0: tensor<16x32x256xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, f]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64
  } : (tensor<16x32x256xf32>, tensor<256x1x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, f]x[o, 0, i]->[b, 0, f]
// CHECK-NOT: transpose

// -----

// CHECK-LABEL: conv1d_ncs_osi_nsc
func.func @conv1d_ncs_osi_nsc(%arg0: tensor<16x256x32xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, f, 0]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64
  } : (tensor<16x256x32xf32>, tensor<256x1x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 1]
// CHECK:      mhlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, f]x[o, 0, i]->[b, 0, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv1d_nsc_sio_nsc
func.func @conv1d_nsc_sio_nsc(%arg0: tensor<16x32x256xf32>, %arg1: tensor<1x256x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64
  } : (tensor<16x32x256xf32>, tensor<1x256x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = "mhlo.transpose"(%arg1)
// CHECK-SAME: permutation
// CHECK-SAME: [2, 0, 1]
// CHECK:      mhlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, f]x[o, 0, i]->[b, 0, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv1d_nsc_osi_ncs
func.func @conv1d_nsc_osi_ncs(%arg0: tensor<16x32x256xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x256x32xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, f]x[o, 0, i]->[b, f, 0]>,
    feature_group_count = 1 : i64
  } : (tensor<16x32x256xf32>, tensor<256x1x256xf32>) -> tensor<16x256x32xf32>
  func.return %0 : tensor<16x256x32xf32>
}

// CHECK-NOT:  transpose
// CHECK:      %[[CONV_OUT:.*]] = mhlo.convolution
// CHECK-SAME: [b, 0, f]x[o, 0, i]->[b, 0, f]
// CHECK:      "mhlo.transpose"(%[[CONV_OUT]])
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 1]


// -----

// CHECK-LABEL: conv1d_ncs_ois_ncs
func.func @conv1d_ncs_ois_ncs(%arg0: tensor<16x256x32xf32>, %arg1: tensor<256x256x1xf32>) -> tensor<16x256x32xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>,
    feature_group_count = 1 : i64
  } : (tensor<16x256x32xf32>, tensor<256x256x1xf32>) -> tensor<16x256x32xf32>
  func.return %0 : tensor<16x256x32xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 1]
// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = "mhlo.transpose"(%arg1)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 1]
// CHECK:      %[[CONV_OUT:.*]] = mhlo.convolution(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, f]x[o, 0, i]->[b, 0, f]
// CHECK:      "mhlo.transpose"(%[[CONV_OUT]])
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 1]


// -----

// 3D
//=--

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc
func.func @conv3d_ndhwc_dhwio_ndhwc(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK-NOT:  transpose
// CHECK:      mhlo.convolution
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ncdhw_dhwio_ndhwc
func.func @conv3d_ncdhw_dhwio_ndhwc(%arg0: tensor<1x207x8x8x32xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, f, 0, 1, 2]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x207x8x8x32xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[TRANSPOSED_INPUT:.*]] = "mhlo.transpose"(%arg0)
// CHECK-SAME: permutation
// CHECK-SAME: [0, 2, 3, 4, 1]
// CHECK:      mhlo.convolution(%[[TRANSPOSED_INPUT]], %arg1)
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ndhwc_odhwi_ndhwc
func.func @conv3d_ndhwc_odhwi_ndhwc(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<16x3x3x32x207xf32>) -> tensor<1x6x6x1x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[o, 0, 1, 2, i]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<16x3x3x32x207xf32>) -> tensor<1x6x6x1x16xf32>
  func.return %0 : tensor<1x6x6x1x16xf32>
}

// CHECK:      %[[TRANSPOSED_KERNEL:.*]] = "mhlo.transpose"(%arg1)
// CHECK-SAME: permutation
// CHECK-SAME: [1, 2, 3, 4, 0]
// CHECK:      mhlo.convolution(%arg0, %[[TRANSPOSED_KERNEL]])
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK-NOT:  transpose

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ncdhw
func.func @conv3d_ndhwc_dhwio_ncdhw(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<1x16x6x6x1xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, f, 0, 1, 2]>,
    feature_group_count = 1 : i64} :
       (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<1x16x6x6x1xf32>
  func.return %0 : tensor<1x16x6x6x1xf32>
}

// CHECK-NOT:  transpose
// CHECK:      %[[CONV_OUT:.*]] = mhlo.convolution
// CHECK-SAME: [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// CHECK:      "mhlo.transpose"(%[[CONV_OUT]])
// CHECK-SAME: permutation
// CHECK-SAME: [0, 4, 1, 2, 3]

// -----

//===----------------------------------------------------------------------===//
// mhlo.pad
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pad_2d
func.func @pad_2d(%arg0: tensor<3x3xf32>, %arg1: tensor<f32>) -> tensor<4x3xf32> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, -1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 1]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<3x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// CHECK: mhlo.slice
// CHECK-SAME: limit_indices = dense<3>
// CHECK-SAME: start_indices = dense<[0, 1]>
// CHECK-SAME: (tensor<3x3xf32>) -> tensor<3x2xf32>
// CHECK: mhlo.pad
// CHECK-SAME: edge_padding_high = dense<1>
// CHECK-SAME: edge_padding_low = dense<0>
// CHECK-SAME: (tensor<3x2xf32>, tensor<f32>) -> tensor<4x3xf32>

// -----

// CHECK-LABEL: pad_2d_negative
func.func @pad_2d_negative(%arg0: tensor<3x3xf32>, %arg1: tensor<f32>) -> tensor<1x2xf32> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[-1, -1]> : tensor<2xi64>,
    edge_padding_high = dense<[-1, 0]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<3x3xf32>, tensor<f32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// CHECK: mhlo.slice
// CHECK-SAME: limit_indices = dense<[2, 3]>
// CHECK-SAME: start_indices = dense<1>
// CHECK-SAME: (tensor<3x3xf32>) -> tensor<1x2xf32>
// CHECK-NOT: mhlo.pad

// -----

// CHECK-LABEL: pad_3d_mixed
func.func @pad_3d_mixed(%arg0: tensor<3x3x3xf32>, %arg1: tensor<f32>) -> tensor<3x3x3xf32> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[-1, 1, 0]> : tensor<3xi64>,
    edge_padding_high = dense<[1, -1, 0]> : tensor<3xi64>,
    interior_padding = dense<0> : tensor<3xi64>
  } : (tensor<3x3x3xf32>, tensor<f32>) -> tensor<3x3x3xf32>
  func.return %0 : tensor<3x3x3xf32>
}

// CHECK: mhlo.slice
// CHECK-SAME: limit_indices = dense<[3, 2, 3]>
// CHECK-SAME: start_indices = dense<[1, 0, 0]>
// CHECK-SAME: (tensor<3x3x3xf32>) -> tensor<2x2x3xf32>
// CHECK: mhlo.pad
// CHECK-SAME: edge_padding_high = dense<[1, 0, 0]>
// CHECK-SAME: edge_padding_low = dense<[0, 1, 0]>
// CHECK-SAME: (tensor<2x2x3xf32>, tensor<f32>) -> tensor<3x3x3xf32>

