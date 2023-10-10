// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

func.func @mhlo_pad_test__noop(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<5x7xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[0, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[0, 0]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<5x7xf32>
  func.return %0: tensor<5x7xf32>

// CHECK-LABEL: mhlo_pad_test__noop
// CHECK: return %arg0 : tensor<5x7xf32>
}

func.func @mhlo_pad_test__pad_all(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<9x10xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[3, 2]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 1]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<9x10xf32>
  func.return %0: tensor<9x10xf32>

// CHECK-LABEL: mhlo_pad_test__pad_all
// CHECK: %cst = arith.constant dense<{{\[}}[3, 1], [2, 1]]> : tensor<2x2xi64>
// CHECK: %0 = "tfl.padv2"(%arg0, %cst, %arg1) : (tensor<5x7xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<9x10xf32>
// CHECK: return %0 : tensor<9x10xf32>
}

func.func @mhlo_pad_test__crop_all(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<3x5xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-1, -1]> : tensor<2xi64>,
    edge_padding_high = dense<[-1, -1]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>

// CHECK-LABEL: mhlo_pad_test__crop_all
// CHECK: %cst = arith.constant dense<1> : tensor<2xi64>
// CHECK: %cst_0 = arith.constant dense<-1> : tensor<2xi64>
// CHECK: %cst_1 = arith.constant dense<1> : tensor<2xi64>
// CHECK: %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<5x7xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x5xf32>
// CHECK: return %0 : tensor<3x5xf32>
}

func.func @mhlo_pad_test__interior_pad_all(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<9x13xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[0, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[0, 0]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<9x13xf32>
  func.return %0: tensor<9x13xf32>

// CHECK-LABEL: mhlo_pad_test__interior_pad_all
// CHECK: %cst = arith.constant dense<2> : tensor<2xi32>
// CHECK: %0 = "tfl.dilate"(%arg0, %cst, %arg1) : (tensor<5x7xf32>, tensor<2xi32>, tensor<f32>) -> tensor<9x13xf32>
// CHECK: return %0 : tensor<9x13xf32>
}

func.func @mhlo_pad_test__pad_and_crop(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<5x7xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-1, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, -1]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<5x7xf32>
  func.return %0: tensor<5x7xf32>

// CHECK-LABEL: mhlo_pad_test__pad_and_crop
// CHECK: %cst = arith.constant dense<{{\[}}[0, 1], [1, 0]]> : tensor<2x2xi64>
// CHECK: %0 = "tfl.padv2"(%arg0, %cst, %arg1) : (tensor<5x7xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<6x8xf32>
// CHECK: %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi64>
// CHECK: %cst_1 = arith.constant dense<[0, -1]> : tensor<2xi64>
// CHECK: %cst_2 = arith.constant dense<1> : tensor<2xi64>
// CHECK: %1 = "tfl.strided_slice"(%0, %cst_0, %cst_1, %cst_2) {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<6x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x7xf32>
// CHECK: return %1 : tensor<5x7xf32>
}

func.func @mhlo_pad_test__pad_and_crop_and_interior_pad(%input: tensor<5x7xf32>, %padding_value: tensor<f32>) -> tensor<13x25xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-1, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, -1]> : tensor<2xi64>,
    interior_padding = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<13x25xf32>
  func.return %0: tensor<13x25xf32>

// CHECK-LABEL: mhlo_pad_test__pad_and_crop_and_interior_pad
// CHECK: %cst = arith.constant dense<[3, 4]> : tensor<2xi32>
// CHECK: %0 = "tfl.dilate"(%arg0, %cst, %arg1) : (tensor<5x7xf32>, tensor<2xi32>, tensor<f32>) -> tensor<13x25xf32>
// CHECK: %cst_0 = arith.constant dense<{{\[}}[0, 1], [1, 0]]> : tensor<2x2xi64>
// CHECK: %1 = "tfl.padv2"(%0, %cst_0, %arg1) : (tensor<13x25xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<14x26xf32>
// CHECK: %cst_1 = arith.constant dense<[1, 0]> : tensor<2xi64>
// CHECK: %cst_2 = arith.constant dense<[0, -1]> : tensor<2xi64>
// CHECK: %cst_3 = arith.constant dense<1> : tensor<2xi64>
// CHECK: %2 = "tfl.strided_slice"(%1, %cst_1, %cst_2, %cst_3) {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<14x26xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<13x25xf32>
// CHECK: return %2 : tensor<13x25xf32>
}

func.func @mhlo_pad_test__pad_all_unknown_shape(%input: tensor<?x?x?x?xf32>, %padding_value: tensor<f32>) -> tensor<?x?x?x?xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    edge_padding_high = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    interior_padding = dense<[0, 0, 0, 0]> : tensor<4xi64>
  } : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  func.return %0: tensor<?x?x?x?xf32>

// CHECK-LABEL: mhlo_pad_test__pad_all_unknown_shape
// CHECK: %cst = arith.constant dense<1> : tensor<4x2xi64>
// CHECK: %0 = "tfl.padv2"(%arg0, %cst, %arg1) : (tensor<?x?x?x?xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK: return %0 : tensor<?x?x?x?xf32>
}

func.func @mhlo_pad_test__crop_all_unknown_shape(%input: tensor<?x?x?x?xf32>, %padding_value: tensor<f32>) -> tensor<?x?x?x?xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-1, -1, -1, -1]> : tensor<4xi64>,
    edge_padding_high = dense<[-1, -1, -1, -1]> : tensor<4xi64>,
    interior_padding = dense<[0, 0, 0, 0]> : tensor<4xi64>
  } : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  func.return %0: tensor<?x?x?x?xf32>

// CHECK-LABEL: mhlo_pad_test__crop_all_unknown_shape
// CHECK: %cst = arith.constant dense<1> : tensor<4xi64>
// CHECK: %cst_0 = arith.constant dense<-1> : tensor<4xi64>
// CHECK: %cst_1 = arith.constant dense<1> : tensor<4xi64>
// CHECK: %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<?x?x?x?xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK: return %0 : tensor<?x?x?x?xf32>
}

func.func @mhlo_pad_test__pad_all_unknown_dim0(%input: tensor<?x2x3x4xf32>, %padding_value: tensor<f32>) -> tensor<?x4x5x6xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    edge_padding_high = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    interior_padding = dense<[0, 0, 0, 0]> : tensor<4xi64>
  } : (tensor<?x2x3x4xf32>, tensor<f32>) -> tensor<?x4x5x6xf32>
  func.return %0: tensor<?x4x5x6xf32>

// CHECK-LABEL: mhlo_pad_test__pad_all_unknown_dim0
// CHECK: %cst = arith.constant dense<1> : tensor<4x2xi64>
// CHECK: %0 = "tfl.padv2"(%arg0, %cst, %arg1) : (tensor<?x2x3x4xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<?x4x5x6xf32>
// CHECK: return %0 : tensor<?x4x5x6xf32>
}

func.func @mhlo_pad_test__crop_all_unknown_dim0(%input: tensor<?x2x3x4xf32>, %padding_value: tensor<f32>) -> tensor<?x0x1x2xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-1, -1, -1, -1]> : tensor<4xi64>,
    edge_padding_high = dense<[-1, -1, -1, -1]> : tensor<4xi64>,
    interior_padding = dense<[0, 0, 0, 0]> : tensor<4xi64>
  } : (tensor<?x2x3x4xf32>, tensor<f32>) -> tensor<?x0x1x2xf32>
  func.return %0: tensor<?x0x1x2xf32>

// CHECK-LABEL: mhlo_pad_test__crop_all_unknown_dim0
// CHECK: %cst = arith.constant dense<1> : tensor<4xi64>
// CHECK: %cst_0 = arith.constant dense<-1> : tensor<4xi64>
// CHECK: %cst_1 = arith.constant dense<1> : tensor<4xi64>
// CHECK: %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<?x2x3x4xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<?x0x1x2xf32>
// CHECK: return %0 : tensor<?x0x1x2xf32>
}

func.func @mhlo_pad_test__pad_and_crop_and_interior_pad_unknown_dim0(%input: tensor<?x2x3x4xf32>, %padding_value: tensor<f32>) -> tensor<?x3x8x15xf32> {
  %0 = "mhlo.pad"(%input, %padding_value) {
    edge_padding_low = dense<[-2, -1, 0, 1]> : tensor<4xi64>,
    edge_padding_high = dense<[1, 0, -1, -2]> : tensor<4xi64>,
    interior_padding = dense<[1, 2, 3, 4]> : tensor<4xi64>
  } : (tensor<?x2x3x4xf32>, tensor<f32>) -> tensor<?x3x8x15xf32>
  func.return %0: tensor<?x3x8x15xf32>

// CHECK-LABEL: mhlo_pad_test__pad_and_crop_and_interior_pad_unknown_dim0
// CHECK: %cst = arith.constant dense<[2, 3, 4, 5]> : tensor<4xi32>
// CHECK: %0 = "tfl.dilate"(%arg0, %cst, %arg1) : (tensor<?x2x3x4xf32>, tensor<4xi32>, tensor<f32>) -> tensor<?x4x9x16xf32>
// CHECK: %cst_0 = arith.constant dense<{{\[}}[0, 1], [0, 0], [0, 0], [1, 0]]> : tensor<4x2xi64>
// CHECK: %1 = "tfl.padv2"(%0, %cst_0, %arg1) : (tensor<?x4x9x16xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<?x4x9x17xf32>
// CHECK: %cst_1 = arith.constant dense<[2, 1, 0, 0]> : tensor<4xi64>
// CHECK: %cst_2 = arith.constant dense<[0, 0, -1, -2]> : tensor<4xi64>
// CHECK: %cst_3 = arith.constant dense<1> : tensor<4xi64>
// CHECK: %2 = "tfl.strided_slice"(%1, %cst_1, %cst_2, %cst_3) {begin_mask = 12 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<?x4x9x17xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<?x3x8x15xf32>
// CHECK: return %2 : tensor<?x3x8x15xf32>
}
