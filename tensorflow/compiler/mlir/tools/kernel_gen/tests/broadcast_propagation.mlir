// RUN: kernel-gen-opt %s --split-input-file --mhlo-broadcast-propagation | \
// RUN: FileCheck %s

// CHECK-LABEL: @single_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @single_bcast(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape : tensor<3xindex>) -> tensor<?x16x?xf32> {
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]])
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE]])
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : [[BCASTED_TY:tensor<\?x16x\?xf32>]]
  // CHECK-DAG: %[[MUL:.*]] = mhlo.multiply %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[DIV:.*]] = mhlo.divide %[[MUL]], %[[SUB]] : [[BCASTED_TY]]
  // CHECK:     return %[[DIV]] : [[BCASTED_TY]]
  %0 = mhlo.add %arg0, %arg1 : tensor<16x?xf32>
  %1 = mhlo.multiply %0, %0 : tensor<16x?xf32>
  %2 = mhlo.subtract %0, %0 : tensor<16x?xf32>
  %3 = mhlo.divide %1, %2 : tensor<16x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  func.return %4 : tensor<?x16x?xf32>
}

// -----

// CHECK-LABEL: @single_bcast_ensure_order
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @single_bcast_ensure_order(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape : tensor<3xindex>) -> tensor<?x16x?xf32> {
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]])
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE]])
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : [[BCASTED_TY:tensor<\?x16x\?xf32>]]
  // CHECK-DAG: %[[MUL:.*]] = mhlo.multiply %[[ADD]], %[[ADD]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[MUL]], %[[MUL]] : [[BCASTED_TY]]
  // CHECK-DAG: %[[DIV:.*]] = mhlo.divide %[[ADD]], %[[SUB]] : [[BCASTED_TY]]
  // CHECK:     return %[[DIV]] : [[BCASTED_TY]]
  %0 = mhlo.add %arg0, %arg1 : tensor<16x?xf32>
  %1 = mhlo.multiply %0, %0 : tensor<16x?xf32>
  %2 = mhlo.subtract %1, %1 : tensor<16x?xf32>
  %3 = mhlo.divide %0, %2 : tensor<16x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %shape) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
      (tensor<16x?xf32>, tensor<3xindex>) -> tensor<?x16x?xf32>
  func.return %4 : tensor<?x16x?xf32>
}

// -----

// CHECK-LABEL: @double_bcasts
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x?xf32>, %[[ARG1:.*]]: tensor<16x?xf32>, %[[SHAPE0:.*]]: tensor<3xindex>, %[[SHAPE1:.*]]: tensor<3xindex>
func.func @double_bcasts(%arg0 : tensor<16x?xf32>, %arg1 : tensor<16x?xf32>,
    %shape0 : tensor<3xindex>, %shape1 : tensor<3xindex>)
    -> (tensor<?x16x?xf32>, tensor<?x16x?xf32>) {
  // CHECK-DAG: %[[BCASTED_ARG00:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE0]]) [[BCAST_DIMS0:<{broadcast_dimensions = dense<\[1, 2\]> : tensor<2xi64>}>]]
  // CHECK-DAG: %[[BCASTED_ARG01:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE1]]) [[BCAST_DIMS1:<{broadcast_dimensions = dense<\[0, 2\]> : tensor<2xi64>}>]]
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
  // CHECK:     return %[[DIV0]], %[[DIV1]] : [[BCAST_TY]], [[BCAST_TY]]
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
  func.return %4, %5 : tensor<?x16x?xf32>, tensor<?x16x?xf32>
}

// -----

// CHECK-LABEL: @late_output_dimensions
// CHECK: %[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>
func.func @late_output_dimensions(%arg0 : tensor<?x32xf32>, %arg1 : tensor<?x32xf32>,
    %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x32xf32> {
  // CHECK-DAG: %[[SHAPE:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]]) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}>
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[SHAPE]]) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}>
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : [[BCAST_TY:tensor<\?x\?x32xf32>]]
  // CHECK-DAG: %[[ADD:.*]] = mhlo.add %[[SUB]], %[[SUB]] : [[BCAST_TY]]
  // CHECK:     return %[[ADD]] : [[BCAST_TY]]
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf32>
  %1 = shape.shape_of %arg2 : tensor<?x?x?xf32> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1)
      {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} :
      (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
  %3 = mhlo.add %2, %2 : tensor<?x?x32xf32>
  func.return %3 : tensor<?x?x32xf32>
}

// -----

// CHECK-LABEL: @very_late_output_dimensions
// CHECK: (%[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
func.func @very_late_output_dimensions(%arg0 : tensor<?x32xf32>,
    %arg1 : tensor<?x32xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x32xf32> {
  // CHECK-DAG: %[[S:.*]] = shape.shape_of %[[ARG2:.*]]
  // CHECK-DAG: %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[S]])
  // CHECK-DAG: %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[S]])
  // CHECK-DAG: %[[ACC0:.*]] = mhlo.add %[[BCASTED_ARG0]], %[[BCASTED_ARG1]]
  // CHECK-DAG: %[[ACC1:.*]] = mhlo.multiply %[[ACC0]], %[[BCASTED_ARG1]]
  // CHECK-DAG: %[[ACC2:.*]] = mhlo.subtract %[[ACC1]], %[[BCASTED_ARG1]]
  // CHECK-DAG: %[[ACC3:.*]] = mhlo.divide %[[ACC2]], %[[BCASTED_ARG1]]
  // CHECK:     return %[[ACC3]]
  %acc0 = mhlo.add %arg0, %arg1 : tensor<?x32xf32>
  %acc1 = mhlo.multiply %acc0, %arg1 : tensor<?x32xf32>
  %acc2 = mhlo.subtract %acc1, %arg1 : tensor<?x32xf32>
  %acc3 = mhlo.divide %acc2, %arg1 : tensor<?x32xf32>
  %1 = shape.shape_of %arg2 : tensor<?x?x?xf32> -> tensor<3xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%acc3, %1) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
  func.return %3 : tensor<?x?x32xf32>
}

// -----

// CHECK-LABEL: @propagate_within_block
// CHECK: (%[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[ARG2:.*]]: tensor<?x32xf32>, %[[W:.*]]: !shape.witness, %[[SHAPE:.*]]: tensor<3xindex>)
func.func @propagate_within_block(%arg0 : tensor<?x32xf32>, %arg1 : tensor<?x32xf32>,
    %arg2 : tensor<?x32xf32>, %w : !shape.witness, %shape : tensor<3xindex>)
    -> tensor<?x?x32xf32> {
  // CHECK-DAG: %[[SUB:.*]] = mhlo.subtract %[[ARG0]], %[[ARG1]] : tensor<?x32xf32>
  // CHECK:     %[[RESULT:.*]] = shape.assuming %[[W]]
  // CHECK-DAG:   %[[BCASTED_ARG2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[SHAPE]])
  // CHECK-DAG:   %[[BCASTED_SUB:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[SUB]], %[[SHAPE]])
  // CHECK-DAG:   %[[ADD:.*]] = mhlo.add %[[BCASTED_SUB]], %[[BCASTED_ARG2]] : tensor<?x?x32xf32>
  // CHECK:       shape.assuming_yield %[[ADD]]
  // CHECK:     return %[[RESULT]]
  %sub = mhlo.subtract %arg0, %arg1 : tensor<?x32xf32>
  %result = shape.assuming %w -> tensor<?x?x32xf32> {
    %add = mhlo.add %sub, %arg2 : tensor<?x32xf32>
    %bcasted = "mhlo.dynamic_broadcast_in_dim"(%add, %shape)
        {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} :
        (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
    shape.assuming_yield %bcasted : tensor<?x?x32xf32>
  }
  func.return %result : tensor<?x?x32xf32>
}

// -----

// CHECK-LABEL: @propagate_within_block_2
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?xf32>, %[[S:.*]]: tensor<3xindex>, %[[W:.*]]: !shape.witness
func.func @propagate_within_block_2(%arg : tensor<?x?x?xf32>,
    %shape: tensor<3xindex>, %w: !shape.witness) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[TMP:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[S]])
  // CHECK:     %[[RES:.*]] = shape.assuming %[[W]]
  // CHECK-DAG:   %[[INNER_RES:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[TMP]], %[[S]])
  // CHECK:       shape.assuming_yield %[[INNER_RES]]
  // CHECK:     return %[[RES]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %1 = shape.assuming %w -> tensor<?x?x?xf32> {
    %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape)
        {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
        : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
    shape.assuming_yield %2 : tensor<?x?x?xf32>
  }
  func.return %1 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @propagate_across_bcasts_cst_src
// CHECK-SAME:  %[[ARG:.*]]: tensor<1xindex>
func.func @propagate_across_bcasts_cst_src(%s : tensor<1xindex>) -> tensor<?xi1> {
  // CHECK-DAG: %[[C1:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK-DAG: %[[RES:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[C1]], %[[ARG]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i1>, tensor<1xindex>) -> tensor<?xi1>
  // CHECK:     return %[[RES]]
  %0 = mhlo.constant dense<true> : tensor<i1>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %s)
      {broadcast_dimensions = dense<> : tensor<0xi64>}
      : (tensor<i1>, tensor<1xindex>) -> tensor<?xi1>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%1, %s)
      {broadcast_dimensions = dense<0> : tensor<1xi64>}
      : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
  func.return %2 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @compose_bcast_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xi1>, %[[S0:.*]]: tensor<3xindex>, %[[S1:.*]]: tensor<4xindex>
func.func @compose_bcast_dims(%arg : tensor<?x?xi1>, %s0 : tensor<3xindex>, %s1 : tensor<4xindex>) -> tensor<1x?x1x?xi1> {
  // CHECK-DAG: %[[RES:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[S1]]) <{broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>}> : (tensor<?x?xi1>, tensor<4xindex>) -> tensor<1x?x1x?xi1>
  // CHECK:     return %[[RES]]
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg, %s0)
      {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
      : (tensor<?x?xi1>, tensor<3xindex>) -> tensor<1x?x?xi1>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%1, %s1)
      {broadcast_dimensions = dense<[0, 1, 3]> : tensor<3xi64>}
      : (tensor<1x?x?xi1>, tensor<4xindex>) -> tensor<1x?x1x?xi1>
  func.return %2 : tensor<1x?x1x?xi1>
}

// -----

// CHECK-LABEL: @propagate_across_bcasts
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?xf32>, %[[S:.*]]: tensor<3xindex>
func.func @propagate_across_bcasts(%arg : tensor<?x?x?xf32>, %shape : tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[RES:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[S]]) <{broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}> : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  // CHECK:     return %[[RES]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%1, %shape)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %2 : tensor<?x?x?xf32>
}
