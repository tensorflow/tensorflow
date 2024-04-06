// RUN: odml-to-stablehlo-opt %s -fuse-mhlo-convolution-pass -cse | FileCheck %s

// CHECK-LABEL: @fuseMulAndConv2D
// CHECK-SAME: %[[INPUT:[^:[:space:]]+]]
func.func @fuseMulAndConv2D(%input: tensor<1x256x256x3xf32>) -> (tensor<1x256x256x2xf32>) {
  // CHECK-DAG: %[[FILTER:.+]] = mhlo.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]]]> : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[CST:.+]] = mhlo.constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
  // CHECK-DAG: %[[CST_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[CST]]) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>) -> tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[NEW_FILTER:.+]] =  mhlo.multiply %[[CST_BCAST]], %[[FILTER]] : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.convolution(%[[INPUT]], %[[NEW_FILTER]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  %filter = mhlo.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x1x3x2xf32>
  %cst = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %0 = mhlo.convolution(%input, %filter) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  %1 = "mhlo.broadcast_in_dim"(%cst) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>) -> tensor<1x256x256x2xf32>
  %2 = mhlo.multiply %0, %1 : tensor<1x256x256x2xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %2 : tensor<1x256x256x2xf32>
}

// -----

// CHECK-LABEL: @fuseMulAndConv2DDynamic
// CHECK-SAME: %[[INPUT:[^:[:space:]]+]]
func.func @fuseMulAndConv2DDynamic(%input: tensor<?x256x256x3xf32>) -> (tensor<?x256x256x2xf32>) {
  // CHECK-DAG: %[[FILTER:.+]] = mhlo.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]]]> : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[CST_0:.+]] = mhlo.constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
  // CHECK-DAG: %[[CST_1:.+]] = mhlo.constant dense<[3.000000e-01, 4.000000e-01]> : tensor<2xf32>
  // CHECK: %[[CST_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[CST_0]]) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>) -> tensor<1x1x3x2xf32>
  // CHECK: %[[NEW_FILTER:.+]] =  mhlo.multiply %[[CST_BCAST]], %[[FILTER]] : tensor<1x1x3x2xf32>
  // CHECK: %[[CONV:.+]] = mhlo.convolution(%[[INPUT]], %[[NEW_FILTER]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<?x256x256x2xf32>
  // CHECK: %[[SHAPE:.+]] = shape.shape_of %[[CONV]] : tensor<?x256x256x2xf32> -> tensor<4xindex>
  // CHECK: %[[DYNAMIC_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[CST_1]], %[[SHAPE]]) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  // CHECK: %[[ADD:.+]] = mhlo.add %[[CONV]], %[[DYNAMIC_BCAST]] : tensor<?x256x256x2xf32>
  %filter = mhlo.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x1x3x2xf32>
  %cst_0 = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %cst_1 = mhlo.constant dense<[0.3, 0.4]> : tensor<2xf32>
  %0 = mhlo.convolution(%input, %filter) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<?x256x256x2xf32>
  %1 = shape.shape_of %0 : tensor<?x256x256x2xf32> -> tensor<4xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%cst_0, %1) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  %3 = mhlo.multiply %0, %2 : tensor<?x256x256x2xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%cst_1, %1) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  %5 = mhlo.add %3, %4 : tensor<?x256x256x2xf32>
  // CHECK-DAG: return %[[ADD]]
  func.return %5 : tensor<?x256x256x2xf32>
}
