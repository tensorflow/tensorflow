// RUN: mlir-hlo-opt %s --split-input-file --mhlo-broadcast-propagation | \
// RUN: FileCheck %s

// CHECK-LABEL: @single_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func @single_bcast(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape : tensor<3xindex>) -> tensor<?x16x?xf32> {
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]])
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE]])
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : [[BCASTED_TY:tensor<\?x16x\?xf32>]]
  // CHECK-DAG: %[[MUL:.*]] = mhlo.multiply %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[DIV:.*]] = mhlo.divide %[[MUL]], %[[SUB]] : [[BCASTED_TY]]
  // CHECK: return %[[DIV]] : [[BCASTED_TY]]
  %0 = mhlo.add %arg0, %arg1 : tensor<16x?xf32>
  %1 = mhlo.multiply %0, %0 : tensor<16x?xf32>
  %2 = mhlo.subtract %0, %0 : tensor<16x?xf32>
  %3 = mhlo.divide %1, %2 : tensor<16x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  return %4 : tensor<?x16x?xf32>
}

// CHECK-LABEL: @single_bcast_ensure_order
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func @single_bcast_ensure_order(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape : tensor<3xindex>) -> tensor<?x16x?xf32> {
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]])
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE]])
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : [[BCASTED_TY:tensor<\?x16x\?xf32>]]
  // CHECK-DAG: %[[MUL:.*]] = mhlo.multiply %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[MUL]], %[[MUL]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[DIV:.*]] = mhlo.divide %[[ADD]], %[[SUB]] : [[BCASTED_TY]]
  // CHECK: return %[[DIV]] : [[BCASTED_TY]]
  %0 = mhlo.add %arg0, %arg1 : tensor<16x?xf32>
  %1 = mhlo.multiply %0, %0 : tensor<16x?xf32>
  %2 = mhlo.subtract %1, %1 : tensor<16x?xf32>
  %3 = mhlo.divide %0, %2 : tensor<16x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  return %4 : tensor<?x16x?xf32>
}

// -----

// CHECK-LABEL: @double_bcasts
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE0:.*]]: tensor<3xindex>, %[[SHAPE1:.*]]: tensor<3xindex>
func @double_bcasts(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape0 : tensor<3xindex>, %shape1 : tensor<3xindex>)
    -> (tensor<?x16x?xf32>, tensor<?x16x?xf32>) {
  // CHECK-DAG: %[[BCASTED_ARG00:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE0]]) [[BCAST_DIMS0:{broadcast_dimensions = dense<\[1, 2\]> : tensor<2xi64>}]]
  // CHECK-DAG: %[[BCASTED_ARG01:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE1]]) [[BCAST_DIMS1:{broadcast_dimensions = dense<\[0, 2\]> : tensor<2xi64>}]]
  // CHECK-DAG: %[[BCASTED_ARG10:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE0]]) [[BCAST_DIMS0]]
  // CHECK-DAG: %[[BCASTED_ARG11:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE1]]) [[BCAST_DIMS1]]
  // CHECK-DAG: %[[ADD0:.*]] = mhlo.add %[[BCASTED_ARG00]], %[[BCASTED_ARG10]] : [[BCAST_TY:tensor<\?x16x\?xf32>]]
  // CHECK-DAG: %[[ADD1:.*]] = mhlo.add %[[BCASTED_ARG01]], %[[BCASTED_ARG11]] : [[BCAST_TY]]
  // CHECK-DAG: %[[MUL0:.*]] = mhlo.multiply %[[ADD0]], %[[ADD0]] : [[BCAST_TY]]
  // CHECK-DAG: %[[MUL1:.*]] = mhlo.multiply %[[ADD1]], %[[ADD1]] : [[BCAST_TY]]
  // CHECK-DAG: %[[SUB0:.*]] = mhlo.subtract %[[ADD0]], %[[ADD0]] : [[BCAST_TY]]
  // CHECK-DAG: %[[SUB1:.*]] = mhlo.subtract %[[ADD1]], %[[ADD1]] : [[BCAST_TY]]
  // CHECK-DAG: %[[DIV0:.*]] = mhlo.divide %[[MUL0]], %[[SUB0]] : [[BCAST_TY]]
  // CHECK-DAG: %[[DIV1:.*]] = mhlo.divide %[[MUL1]], %[[SUB1]] : [[BCAST_TY]]
  // CHECK: return %[[DIV0]], %[[DIV1]] : [[BCAST_TY]], [[BCAST_TY]]
  %0 = mhlo.add %arg0, %arg1 : tensor<16x?xf32>
  %1 = mhlo.multiply %0, %0 : tensor<16x?xf32>
  %2 = mhlo.subtract %0, %0 : tensor<16x?xf32>
  %3 = mhlo.divide %1, %2 : tensor<16x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape0) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape1) {
      broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  return %4, %5 : tensor<?x16x?xf32>, tensor<?x16x?xf32>
}

// -----

// CHECK-LABEL: @late_output_dimensions
// CHECK: %[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>
func @late_output_dimensions(%arg0: tensor<?x32xf32>, %arg1: tensor<?x32xf32>,
    %arg2: tensor<?x?x?xf32>) -> tensor<?x?x32xf32> {
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[ARG0]], %[[ARG1]] : tensor<?x32xf32>
  // CHECK-DAG: %[[SHAPE:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[BCASTED_SUB:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[SUB]], %[[SHAPE]])
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[BCASTED_SUB]], %[[BCASTED_SUB]] : tensor<?x?x32xf32>
  // CHECK-DAG: return %[[ADD]]
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf32>
  %1 = shape.shape_of %arg2 : tensor<?x?x?xf32> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
  %3 = mhlo.add %2, %2 : tensor<?x?x32xf32>
  return %3 : tensor<?x?x32xf32>
}
