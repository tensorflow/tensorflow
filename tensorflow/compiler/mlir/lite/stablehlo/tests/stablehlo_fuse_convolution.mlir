// RUN: odml-to-stablehlo-opt %s -stablehlo-fuse-convolution -cse | FileCheck %s

// CHECK-LABEL: @fuseMulAndConv2D
// CHECK-SAME: %[[INPUT:[^:[:space:]]+]]
func.func @fuseMulAndConv2D(%input: tensor<1x256x256x3xf32>) -> (tensor<1x256x256x2xf32>) {
  // CHECK-DAG: %[[FILTER:.+]] = stablehlo.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]]]> : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[CST:.+]] = stablehlo.constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
  // CHECK-DAG: %[[CST_BCAST:.+]] = stablehlo.broadcast_in_dim %[[CST]], dims = [3] : (tensor<2xf32>) -> tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[NEW_FILTER:.+]] =  stablehlo.multiply %[[FILTER]], %[[CST_BCAST]] : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[RESULT:.+]] = stablehlo.convolution(%[[INPUT]], %[[NEW_FILTER]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  %filter = stablehlo.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x1x3x2xf32>
  %cst = stablehlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %0 = stablehlo.convolution(%input, %filter) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x256x256x2xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [3] : (tensor<2xf32>) -> tensor<1x256x256x2xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<1x256x256x2xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %2 : tensor<1x256x256x2xf32>
}

// -----

// CHECK-LABEL: @fuseMulAndConv2DDynamic
// CHECK-SAME: %[[INPUT:[^:[:space:]]+]]
func.func @fuseMulAndConv2DDynamic(%input: tensor<?x256x256x3xf32>) -> (tensor<?x256x256x2xf32>) {
  // CHECK-DAG: %[[FILTER:.+]] = stablehlo.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]]]> : tensor<1x1x3x2xf32>
  // CHECK-DAG: %[[CST_0:.+]] = stablehlo.constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
  // CHECK-DAG: %[[CST_1:.+]] = stablehlo.constant dense<[3.000000e-01, 4.000000e-01]> : tensor<2xf32>
  // CHECK: %[[CST_BCAST:.+]] = stablehlo.broadcast_in_dim %[[CST_0]], dims = [3] : (tensor<2xf32>) -> tensor<1x1x3x2xf32>
  // CHECK: %[[NEW_FILTER:.+]] =  stablehlo.multiply %[[FILTER]], %[[CST_BCAST]] : tensor<1x1x3x2xf32>
  // CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[INPUT]], %[[NEW_FILTER]]) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<?x256x256x2xf32>
  // CHECK: %[[SHAPE:.+]] = shape.shape_of %[[CONV]] : tensor<?x256x256x2xf32> -> tensor<4xindex>
  // CHECK: %[[DYNAMIC_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[CST_1]], %[[SHAPE]], dims = [3] : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  // CHECK: %[[ADD:.+]] = stablehlo.add %[[CONV]], %[[DYNAMIC_BCAST]] : tensor<?x256x256x2xf32>
  %filter = stablehlo.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x1x3x2xf32>
  %cst_0 = stablehlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %cst_1 = stablehlo.constant dense<[0.3, 0.4]> : tensor<2xf32>
  %0 = stablehlo.convolution(%input, %filter) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x256x256x3xf32>, tensor<1x1x3x2xf32>) -> tensor<?x256x256x2xf32>
  %1 = shape.shape_of %0 : tensor<?x256x256x2xf32> -> tensor<4xindex>
  %2 = stablehlo.dynamic_broadcast_in_dim %cst_0, %1, dims = [3] : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  %3 = stablehlo.multiply %0, %2 : tensor<?x256x256x2xf32>
  %4 = stablehlo.dynamic_broadcast_in_dim %cst_1, %1, dims = [3] : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x256x256x2xf32>
  %5 = stablehlo.add %3, %4 : tensor<?x256x256x2xf32>
  // CHECK-DAG: return %[[ADD]]
  func.return %5 : tensor<?x256x256x2xf32>
}
