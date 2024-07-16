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

// TODO: b/351437662 - Add support for non-standard layouts.
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

// CHECK-NOT: transpose
// CHECK:     [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
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

// CHECK-NOT: transpose
// CHECK:     [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
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

// CHECK-NOT: transpose
// CHECK:     [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
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

// CHECK-NOT: transpose
// CHECK:     [b, 0, 1, f]x[o, 0, 1, i]->[b, f, 0, 1]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
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

// CHECK-NOT: transpose
// CHECK:     [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
// CHECK-NOT: transpose

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

// TODO: b/351437662 - Add support for non-standard layouts.
// CHECK-LABEL: conv1d_ncs_osi_nsc
func.func @conv1d_ncs_osi_nsc(%arg0: tensor<16x256x32xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, f, 0]x[o, 0, i]->[b, 0, f]>,
    feature_group_count = 1 : i64
  } : (tensor<16x256x32xf32>, tensor<256x1x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, f, 0]x[o, 0, i]->[b, 0, f]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
// CHECK-LABEL: conv1d_nsc_sio_nsc
func.func @conv1d_nsc_sio_nsc(%arg0: tensor<16x32x256xf32>, %arg1: tensor<1x256x256xf32>) -> tensor<16x32x256xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64
  } : (tensor<16x32x256xf32>, tensor<1x256x256xf32>) -> tensor<16x32x256xf32>
  func.return %0 : tensor<16x32x256xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, f]x[0, i, o]->[b, 0, f]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
// CHECK-LABEL: conv1d_nsc_osi_ncs
func.func @conv1d_nsc_osi_ncs(%arg0: tensor<16x32x256xf32>, %arg1: tensor<256x1x256xf32>) -> tensor<16x256x32xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, f]x[o, 0, i]->[b, f, 0]>,
    feature_group_count = 1 : i64
  } : (tensor<16x32x256xf32>, tensor<256x1x256xf32>) -> tensor<16x256x32xf32>
  func.return %0 : tensor<16x256x32xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, 0, f]x[o, 0, i]->[b, f, 0]
// CHECK-NOT: transpose

// -----

// TODO: b/351437662 - Add support for non-standard layouts.
// CHECK-LABEL: conv1d_ncs_ois_ncs
func.func @conv1d_ncs_ois_ncs(%arg0: tensor<16x256x32xf32>, %arg1: tensor<256x256x1xf32>) -> tensor<16x256x32xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>,
    feature_group_count = 1 : i64
  } : (tensor<16x256x32xf32>, tensor<256x256x1xf32>) -> tensor<16x256x32xf32>
  func.return %0 : tensor<16x256x32xf32>
}

// CHECK-NOT: transpose
// CHECK:     [b, f, 0]x[o, i, 0]->[b, f, 0]
// CHECK-NOT: transpose



