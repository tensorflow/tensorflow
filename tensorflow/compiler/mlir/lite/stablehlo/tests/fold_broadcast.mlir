// RUN: odml-to-stablehlo-opt %s -constant-fold-broadcast-pass -cse -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_float
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_float() -> (tensor<1x1x2x4xf32>) {
  // CHECK-DAG: %[[RES:.*]] = mhlo.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00, 9.000000e+00, 1.600000e+01], [5.000000e+00, 1.200000e+01, 2.100000e+01, 3.200000e+01]]]]> : tensor<1x1x2x4xf32>
  %cst0 = mhlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst1 = mhlo.constant dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]> : tensor<1x1x2x4xf32>
  %0 = "mhlo.broadcast_in_dim"(%cst0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x1x2x4xf32>
  %1 = mhlo.multiply %0, %cst1 : tensor<1x1x2x4xf32>
  // CHECK:      return %[[RES]] : tensor<1x1x2x4xf32>
  func.return %1 : tensor<1x1x2x4xf32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_2D_float
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_2D_float() -> (tensor<1x2x2x3xf32>) {
  // CHECK-DAG: %[[RES:.*]] = mhlo.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00, 9.000000e+00], [4.000000e+00, 1.000000e+01, 1.800000e+01]], {{\[\[}}2.800000e+01, 4.000000e+01, 5.400000e+01], [4.000000e+01, 5.500000e+01, 7.200000e+01]]]]> : tensor<1x2x2x3xf32>
  %cst0 = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %cst1 = mhlo.constant dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]> : tensor<1x2x2x3xf32>
  %0 = "mhlo.broadcast_in_dim"(%cst0) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<1x2x2x3xf32>
  %1 = mhlo.multiply %0, %cst1 : tensor<1x2x2x3xf32>
  // CHECK:      return %[[RES]] : tensor<1x2x2x3xf32>
  func.return %1 : tensor<1x2x2x3xf32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_int
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_1D_int() -> (tensor<1x1x2x4xi32>) {
  // CHECK-DAG: %[[RES:.*]] = mhlo.constant dense<{{\[\[\[\[}}1, 4, 9, 16], [5, 12, 21, 32]]]]> : tensor<1x1x2x4xi32>
  %cst0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %cst1 = mhlo.constant dense<[[[[1, 2, 3, 4], [5, 6, 7, 8]]]]> : tensor<1x1x2x4xi32>
  %0 = "mhlo.broadcast_in_dim"(%cst0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1x1x2x4xi32>
  %1 = mhlo.multiply %0, %cst1 : tensor<1x1x2x4xi32>
  // CHECK:      return %[[RES]] : tensor<1x1x2x4xi32>
  func.return %1 : tensor<1x1x2x4xi32>
}

// CHECK-LABEL: @foldBroadcastInDimBeforeMulOp_bcast_dim_4D_int
func.func @foldBroadcastInDimBeforeMulOp_bcast_dim_4D_int(%arg0: tensor<1x2x1x4xi32>) -> tensor<1x2x1x4xi32> {
  // CHECK-DAG: %[[RES:.*]] = mhlo.constant dense<{{\[\[\[\[}}0, 1, 2, 3]], {{\[\[}}0, 1, 2, 3]]]]> : tensor<1x2x1x4xi32>
  %0 = mhlo.constant dense<[[[[0, 1, 2, 3]]]]> : tensor<1x1x1x4xi32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x4xi32>) -> tensor<1x2x1x4xi32>
  // CHECK: mhlo.multiply %[[ARG0:.*]], %[[RES]] : tensor<1x2x1x4xi32>
  %2 = mhlo.multiply %arg0, %1 : tensor<1x2x1x4xi32>
  return %2 : tensor<1x2x1x4xi32>
}
