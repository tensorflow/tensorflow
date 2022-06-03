// RUN: mlir-hlo-opt %s --gml-fusion | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>) -> tensor<3x4x5xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
  // CHECK-DAG: %[[RESULT_SPACE:.*]] = gml_st.space [123, 456, 789] : !gml_st.tile<123x456x789>
  // CHECK-DAG: %[[RESULT_TILE:.*]] = gml_st.tile %[[RESULT_SPACE]] [0, 1, 2] [3, 4, 5] [1, 1, 1] : !gml_st.tile<123x456x789> to !gml_st.tile<3x4x5>

  // Check materialization of operand space.
  // CHECK-DAG: %[[OPERAND_SPACE_D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK-DAG: %[[OPERAND_SPACE_D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK-DAG: %[[OPERAND_SPACE:.*]] = gml_st.space [%[[OPERAND_SPACE_D0]], %[[OPERAND_SPACE_D1]]] : !gml_st.tile<?x?>

  // Check materialization of broadcast for subset of interest.
  // CHECK-DAG: %[[OPERAND_TILE:.*]] = gml_st.operand_tile_for_dynamic_broadcast_in_dim %[[RESULT_TILE]], %[[OPERAND_SPACE]], %[[RESULT_SPACE]], [0, 2] : !gml_st.tile<3x4x5>, !gml_st.tile<?x?>, !gml_st.tile<123x456x789> -> !gml_st.tile<?x?>
  // CHECK-DAG: %[[SUB_ARG:.*]] = gml_st.materialize %[[ARG]] at %[[OPERAND_TILE]] : tensor<?x?xf32> at !gml_st.tile<?x?>
  // CHECK-DAG: %[[SUB_INIT:.*]] = gml_st.materialize %[[INIT]] at %[[RESULT_TILE]] : tensor<?x?x?xf32> at !gml_st.tile<3x4x5>
  // CHECK-DAG: %[[SUB_RESULT:.*]] = gml_st.dynamic_broadcast_in_dim %[[SUB_INIT]], %[[SUB_ARG]], [0, 2] : tensor<3x4x5xf32>, tensor<?x?xf32> -> tensor<3x4x5xf32>
  // CHECK:     return %[[SUB_RESULT]]

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // `mhlo.dynamic_broadcast_in_dim` lowered to dst-style op.
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = gml_st.dynamic_broadcast_in_dim %dst, %arg, [0, 2]
      : tensor<?x?x?xf32>, tensor<?x?xf32> -> tensor<?x?x?xf32>

  // Materialze a tile.
  %space = gml_st.space [123, 456, 789] : !gml_st.tile<123x456x789>
  %tile = gml_st.tile %space [0, 1, 2] [3, 4, 5] [1, 1, 1]
      : !gml_st.tile<123x456x789> to !gml_st.tile<3x4x5>
  %bcast_sub = gml_st.materialize %bcast at %tile
      : tensor<?x?x?xf32> at !gml_st.tile<3x4x5>

  func.return %bcast_sub : tensor<3x4x5xf32>
}


// CHECK-LABEL: @add
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @add(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]] at %[[TILE]] : tensor<32x32xf32> at !gml_st.tile<?x?>
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]] at %[[TILE]] : tensor<32x32xf32> at !gml_st.tile<?x?>
  // CHECK-DAG: %[[RES:.*]] = mhlo.add %[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>
  // CHECK:     return %[[RES]]
  %0 = mhlo.add %lhs, %rhs : tensor<32x32xf32>
  %1 = gml_st.materialize %0 at %tile : tensor<32x32xf32> at !gml_st.tile<?x?>
  func.return %1 : tensor<?x?xf32>
}
