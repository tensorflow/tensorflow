// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-fusion="producer-label=producer consumer-label=consumer" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim_at_tile
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<3xindex>, %[[ARG2:.*]]: !gml_st.tile<3x4x?>
func.func @dynamic_broadcast_in_dim_at_tile(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>, %tile : !gml_st.tile<3x4x?>)
    -> tensor<3x4x?xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C2:.*]] = arith.constant 2
  // CHECK:      %[[EXTRACT:.*]] = tensor.extract %[[ARG1]][%[[C0]]]
  // CHECK:      %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]][%[[C1]]]
  // CHECK:      %[[EXTRACT_1:.*]] = tensor.extract %[[ARG1]][%[[C2]]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[EXTRACT]], %[[EXTRACT_0]], %[[EXTRACT_1]]]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[OFFSET_1:.*]] = gml_st.offset %[[ARG2]][%[[C2]]]
  // CHECK:      %[[SIZE_1:.*]] = gml_st.size %[[ARG2]][%[[C2]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [%[[EXTRACT]], %[[EXTRACT_0]], %[[EXTRACT_1]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]], %[[OFFSET_1]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]], %[[SIZE_1]]]
  // CHECK-SAME:     [1, 1, 1]
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[DROP:.*]] = gml_st.drop_dims %[[TILE]], [0, 2]
  // CHECK:      %[[CMPI:.*]] = arith.cmpi ne, %[[DIM]], %[[EXTRACT]]
  // CHECK:      %[[CMPI_0:.*]] = arith.cmpi ne, %[[DIM_0]], %[[EXTRACT_1]]
  // CHECK:      %[[OFFSET_2:.*]] = gml_st.offset %[[DROP]][%[[C0]]]
  // CHECK:      %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[OFFSET_2]]
  // CHECK:      %[[OFFSET_3:.*]] = gml_st.offset %[[DROP]][%[[C1]]]
  // CHECK:      %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[OFFSET_3]]
  // CHECK:      %[[SIZE_2:.*]] = gml_st.size %[[DROP]][%[[C0]]]
  // CHECK:      %[[SELECT_1:.*]] = arith.select %[[CMPI]], %[[C1]], %[[SIZE_2]]
  // CHECK:      %[[SIZE_3:.*]] = gml_st.size %[[DROP]][%[[C1]]]
  // CHECK:      %[[SELECT_2:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[SIZE_3]]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[SELECT]], %[[SELECT_0]]]
  // CHECK-SAME:     [%[[SELECT_1]], %[[SELECT_2]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG0]][%[[TILE_0]]]
  // CHECK:      %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
  // CHECK-SAME:     ins(%[[MATERIALIZE_0]] : tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE]] : tensor<?x?x?xf32>)
  // CHECK-SAME:     broadcast_dimensions = [0, 2]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[DYNAMIC]]
  // CHECK:      return {op_label = "consumer"} %[[CAST]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = thlo.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      broadcast_dimensions = [0, 2]
      { op_label = "producer" }
  %bcast_sub = gml_st.materialize %bcast[%tile]
      : tensor<?x?x?xf32>[!gml_st.tile<3x4x?>]
  func.return { op_label = "consumer" } %bcast_sub : tensor<3x4x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_broadcast_in_dim_at_point
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<3xindex>, %[[ARG2:.*]]: !gml_st.tile<1x1x1>
func.func @dynamic_broadcast_in_dim_at_point(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>, %point : !gml_st.tile<1x1x1>)
    -> tensor<1x1x1xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C2:.*]] = arith.constant 2
  // CHECK:      %[[EXTRACT:.*]] = tensor.extract %[[ARG1]][%[[C0]]]
  // CHECK:      %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]][%[[C1]]]
  // CHECK:      %[[EXTRACT_1:.*]] = tensor.extract %[[ARG1]][%[[C2]]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[EXTRACT]], %[[EXTRACT_0]], %[[EXTRACT_1]]]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[OFFSET_1:.*]] = gml_st.offset %[[ARG2]][%[[C2]]]
  // CHECK:      %[[SIZE_1:.*]] = gml_st.size %[[ARG2]][%[[C2]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [%[[EXTRACT]], %[[EXTRACT_0]], %[[EXTRACT_1]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]], %[[OFFSET_1]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]], %[[SIZE_1]]]
  // CHECK-SAME:     [1, 1, 1]
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[DROP:.*]] = gml_st.drop_dims %[[TILE]], [0, 2]
  // CHECK:      %[[CMPI:.*]] = arith.cmpi ne, %[[DIM]], %[[EXTRACT]]
  // CHECK:      %[[CMPI_0:.*]] = arith.cmpi ne, %[[DIM_0]], %[[EXTRACT_1]]
  // CHECK:      %[[OFFSET_2:.*]] = gml_st.offset %[[DROP]][%[[C0]]]
  // CHECK:      %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[OFFSET_2]]
  // CHECK:      %[[OFFSET_3:.*]] = gml_st.offset %[[DROP]][%[[C1]]]
  // CHECK:      %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[OFFSET_3]]
  // CHECK:      %[[SIZE_2:.*]] = gml_st.size %[[DROP]][%[[C0]]]
  // CHECK:      %[[SELECT_1:.*]] = arith.select %[[CMPI]], %[[C1]], %[[SIZE_2]]
  // CHECK:      %[[SIZE_3:.*]] = gml_st.size %[[DROP]][%[[C1]]]
  // CHECK:      %[[SELECT_2:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[SIZE_3]]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[SELECT]], %[[SELECT_0]]]
  // CHECK-SAME:     [%[[SELECT_1]], %[[SELECT_2]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG0]][%[[TILE_0]]]
  // CHECK:      %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
  // CHECK-SAME:     ins(%[[MATERIALIZE_0]] : tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE]] : tensor<?x?x?xf32>)
  // CHECK-SAME:     broadcast_dimensions = [0, 2]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[DYNAMIC]]
  // CHECK:      return {op_label = "consumer"} %[[CAST]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = thlo.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      broadcast_dimensions = [0, 2]
      { op_label = "producer" }
  %bcast_sub = gml_st.materialize %bcast[%point]
      : tensor<?x?x?xf32>[!gml_st.tile<1x1x1>]
  func.return { op_label = "consumer" } %bcast_sub : tensor<1x1x1xf32>
}

// -----

// CHECK-LABEL: @concatenate_at_tile
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>, %[[ARG3:.*]]: tensor<?x?xi32>, %[[ARG4:.*]]: !gml_st.tile<?x?>
func.func @concatenate_at_tile(%init : tensor<?x?xi32>, %a: tensor<?x?xi32>,
    %b: tensor<?x?xi32>, %c: tensor<?x?xi32>, %tile : !gml_st.tile<?x?>)
    -> tensor<?x?xi32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG4]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG4]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG4]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG4]][%[[C1]]]
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[DIM_1:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  // CHECK:      %[[DIM_2:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [%[[DIM_1]], %[[DIM_2]]]
  // CHECK:      %[[MINUI:.*]] = arith.minui %[[OFFSET_0]], %[[DIM_2]]
  // CHECK:      %[[SUBI:.*]] = arith.subi %[[DIM_2]], %[[MINUI]]
  // CHECK:      %[[MINUI_0:.*]] = arith.minui %[[SUBI]], %[[SIZE_0]]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[OFFSET]], %[[MINUI]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_0]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
  // CHECK:      %[[CMPI:.*]] = arith.cmpi ule, %[[OFFSET_0]], %[[DIM_2]]
  // CHECK:      %[[SUBI_0:.*]] = arith.subi %[[OFFSET_0]], %[[DIM_2]]
  // CHECK:      %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[SUBI_0]]
  // CHECK:      %[[DIM_3:.*]] = tensor.dim %[[ARG2]], %[[C1]]
  // CHECK:      %[[SPACE_1:.*]] = gml_st.space [%[[DIM_1]], %[[DIM_3]]]
  // CHECK:      %[[MINUI_1:.*]] = arith.minui %[[SELECT]], %[[DIM_3]]
  // CHECK:      %[[SUBI_1:.*]] = arith.subi %[[DIM_3]], %[[MINUI_1]]
  // CHECK:      %[[MINUI_2:.*]] = arith.minui %[[SUBI_1]], %[[SIZE_0]]
  // CHECK:      %[[TILE_1:.*]] = gml_st.tile
  // CHECK-SAME:     %[[SPACE_1]] [%[[OFFSET]], %[[MINUI_1]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_2]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG2]][%[[TILE_1]]]
  // CHECK:      %[[CMPI_0:.*]] = arith.cmpi ule, %[[SELECT]], %[[DIM_3]]
  // CHECK:      %[[SUBI_2:.*]] = arith.subi %[[SELECT]], %[[DIM_3]]
  // CHECK:      %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[SUBI_2]]
  // CHECK:      %[[DIM_4:.*]] = tensor.dim %[[ARG3]], %[[C1]]
  // CHECK:      %[[SPACE_2:.*]] = gml_st.space [%[[DIM_1]], %[[DIM_4]]]
  // CHECK:      %[[MINUI_3:.*]] = arith.minui %[[SELECT_0]], %[[DIM_4]]
  // CHECK:      %[[SUBI_3:.*]] = arith.subi %[[DIM_4]], %[[MINUI_3]]
  // CHECK:      %[[MINUI_4:.*]] = arith.minui %[[SUBI_3]], %[[SIZE_0]]
  // CHECK:      %[[TILE_2:.*]] = gml_st.tile %[[SPACE_2]]
  // CHECK-SAME:     [%[[OFFSET]], %[[MINUI_3]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_4]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG3]][%[[TILE_2]]]
  // CHECK:      %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[CONCATENATE:.*]] = thlo.concatenate
  // CHECK-SAME:     ins(%[[MATERIALIZE]] : tensor<?x?xi32>, %[[MATERIALIZE_0]] : tensor<?x?xi32>, %[[MATERIALIZE_1]] : tensor<?x?xi32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_2]] : tensor<?x?xi32>)
  // CHECK-SAME:     dimension = 1
  // CHECK:      return {op_label = "consumer"} %[[CONCATENATE]]
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>) {
      dimension = 1 : i64,
      op_label = "producer" }
  %concat_sub = gml_st.materialize %concat[%tile]
      : tensor<?x?xi32>[!gml_st.tile<?x?>]
  func.return { op_label = "consumer" } %concat_sub : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @concatenate_at_point
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>, %[[ARG3:.*]]: !gml_st.tile<1x1>
func.func @concatenate_at_point(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>,
    %c: tensor<?x?xi32>, %point : !gml_st.tile<1x1>) -> tensor<1x1xi32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[DIM_1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  // CHECK:      %[[DIM_2:.*]] = tensor.dim %[[ARG2]], %[[C1]]
  // CHECK:      %[[ADDI:.*]] = arith.addi %[[DIM_0]], %[[DIM_1]]
  // CHECK:      %[[ADDI_0:.*]] = arith.addi %[[ADDI]], %[[DIM_2]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[DIM]], %[[ADDI_0]]]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG3]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG3]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG3]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG3]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [%[[DIM]], %[[ADDI_0]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[MINUI:.*]] = arith.minui %[[OFFSET_0]], %[[DIM_0]]
  // CHECK:      %[[SUBI:.*]] = arith.subi %[[DIM_0]], %[[MINUI]]
  // CHECK:      %[[MINUI_0:.*]] = arith.minui %[[SUBI]], %[[SIZE_0]]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[OFFSET]], %[[MINUI]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_0]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE_0]]]
  // CHECK:      %[[CMPI:.*]] = arith.cmpi ule, %[[OFFSET_0]], %[[DIM_0]]
  // CHECK:      %[[SUBI_0:.*]] = arith.subi %[[OFFSET_0]], %[[DIM_0]]
  // CHECK:      %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[SUBI_0]]
  // CHECK:      %[[SPACE_1:.*]] = gml_st.space [%[[DIM]], %[[DIM_1]]]
  // CHECK:      %[[MINUI_1:.*]] = arith.minui %[[SELECT]], %[[DIM_1]]
  // CHECK:      %[[SUBI_1:.*]] = arith.subi %[[DIM_1]], %[[MINUI_1]]
  // CHECK:      %[[MINUI_2:.*]] = arith.minui %[[SUBI_1]], %[[SIZE_0]]
  // CHECK:      %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]]
  // CHECK-SAME:     [%[[OFFSET]], %[[MINUI_1]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_2]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_1]]]
  // CHECK:      %[[CMPI_0:.*]] = arith.cmpi ule, %[[SELECT]], %[[DIM_1]]
  // CHECK:      %[[SUBI_2:.*]] = arith.subi %[[SELECT]], %[[DIM_1]]
  // CHECK:      %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[SUBI_2]]
  // CHECK:      %[[SPACE_2:.*]] = gml_st.space [%[[DIM]], %[[DIM_2]]]
  // CHECK:      %[[MINUI_3:.*]] = arith.minui %[[SELECT_0]], %[[DIM_2]]
  // CHECK:      %[[SUBI_3:.*]] = arith.subi %[[DIM_2]], %[[MINUI_3]]
  // CHECK:      %[[MINUI_4:.*]] = arith.minui %[[SUBI_3]], %[[SIZE_0]]
  // CHECK:      %[[TILE_2:.*]] = gml_st.tile %[[SPACE_2]]
  // CHECK-SAME:     [%[[OFFSET]], %[[MINUI_3]]]
  // CHECK-SAME:     [%[[SIZE]], %[[MINUI_4]]]
  // CHECK-SAME:     [%[[C1]], %[[C1]]]
  // CHECK:      %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG2]][%[[TILE_2]]]
  // CHECK:      %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[CONCATENATE:.*]] = thlo.concatenate
  // CHECK-SAME:     ins(%[[MATERIALIZE]] : tensor<?x?xi32>, %[[MATERIALIZE_0]] : tensor<?x?xi32>, %[[MATERIALIZE_1]] : tensor<?x?xi32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_2]] : tensor<?x?xi32>)
  // CHECK-SAME:     dimension = 1
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[CONCATENATE]]
  // CHECK:      return {op_label = "consumer"} %[[CAST]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %a, %c0 : tensor<?x?xi32>
  %concat_dim_a = tensor.dim %a, %c1 : tensor<?x?xi32>
  %concat_dim_b = tensor.dim %b, %c1 : tensor<?x?xi32>
  %concat_dim_c = tensor.dim %c, %c1 : tensor<?x?xi32>
  %concat_dim_ab = arith.addi %concat_dim_a, %concat_dim_b : index
  %concat_dim_abc = arith.addi %concat_dim_ab, %concat_dim_c : index
  %init = linalg.init_tensor [%dim_0, %concat_dim_abc] : tensor<?x?xi32>
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>) {
      dimension = 1 : i64,
      op_label = "producer" }
  %concat_sub = gml_st.materialize %concat[%point]
      : tensor<?x?xi32>[!gml_st.tile<1x1>]
  func.return { op_label = "consumer" } %concat_sub : tensor<1x1xi32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x32xf32>, %[[ARG1:.*]]: tensor<32x32xf32>, %[[ARG2:.*]]: !gml_st.tile<?x?>
func.func @add(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [32, 32]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32):
  // CHECK:        %[[ADDF:.*]] = arith.addf %[[ARG3]], %[[ARG4]]
  // CHECK:        linalg.yield %[[ADDF]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  return { op_label = "consumer" } %result : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_point
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x32xf32>, %[[ARG1:.*]]: tensor<32x32xf32>, %[[ARG2:.*]]: !gml_st.tile<1x1>
func.func @add_point(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %point: !gml_st.tile<1x1>) -> tensor<1x1xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [32, 32]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_1]]
  // CHECK:      %[[CAST_0:.*]] = tensor.cast %[[MATERIALIZE]]
  // CHECK:      %[[CAST_1:.*]] = tensor.cast %[[MATERIALIZE_0]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[CAST_0]], %[[CAST_1]] : tensor<1x1xf32>, tensor<1x1xf32>)
  // CHECK-SAME:     outs(%[[CAST]] : tensor<1x1xf32>)
  // CHECK:      ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32):
  // CHECK:        %[[ADDF:.*]] = arith.addf %[[ARG3]], %[[ARG4]]
  // CHECK:        linalg.yield %[[ADDF]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer"}
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%point]
      : tensor<32x32xf32>[!gml_st.tile<1x1>]
  return { op_label = "consumer" } %result : tensor<1x1xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_two_users
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x32xf32>, %[[ARG1:.*]]: tensor<32x32xf32>, %[[ARG2:.*]]: !gml_st.tile<?x?>, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index
func.func @add_two_users(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [32, 32]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
  // CHECK:        %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
  // CHECK:        linalg.yield %[[ADDF]]
  // CHECK:      %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG5_0:.*]]: f32, %[[ARG6_0:.*]]: f32, %[[ARG7_0:.*]]: f32):
  // CHECK:        %[[ADDF_0:.*]] = arith.addf %[[ARG5_0]], %[[ARG6_0]]
  // CHECK:        linalg.yield %[[ADDF_0]]
  // CHECK:      %[[INIT_0:.*]] = linalg.init_tensor [%[[ARG3]], %[[ARG4]]]
  // CHECK:      %[[GENERIC_1:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[GENERIC]], %[[GENERIC_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_0]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG5_1:.*]]: f32, %[[ARG6_1:.*]]: f32, %[[ARG7_1:.*]]: f32):
  // CHECK:        %[[ADDF_1:.*]] = arith.addf %[[ARG5_1]], %[[ARG6_1]]
  // CHECK:        linalg.yield %[[ADDF_1]]
  // CHECK:      return %[[GENERIC_1]]
  %init0 = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg0 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init0 : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %user0 = gml_st.materialize %linalg0[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %user1 = gml_st.materialize %linalg0[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %init1 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %linalg1 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "consumer" }
      ins(%user0, %user1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init1 : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>
  func.return %linalg1 : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @cos
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x32xf32>, %[[ARG1:.*]]: !gml_st.tile<?x?>
func.func @cos(%arg: tensor<32x32xf32>, %tile: !gml_st.tile<?x?>)
    -> tensor<?x?xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG1]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG1]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [32, 32]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[MATERIALIZE]] : tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_0]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK:        %[[COS:.*]] = math.cos %[[ARG2]]
  // CHECK:        linalg.yield %[[COS]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%arg : tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%arg_scalar: f32, %_: f32):
    %cos = math.cos %arg_scalar : f32
    linalg.yield %cos : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  return { op_label = "consumer" } %result : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @cos_point
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x32xf32>, %[[ARG1:.*]]: !gml_st.tile<1x1>
func.func @cos_point(%arg: tensor<32x32xf32>, %point: !gml_st.tile<1x1>)
    -> tensor<1x1xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG1]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG1]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [32, 32]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_0]]
  // CHECK:      %[[CAST_0:.*]] = tensor.cast %[[MATERIALIZE]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[CAST_0]] : tensor<1x1xf32>)
  // CHECK-SAME:     outs(%[[CAST]] : tensor<1x1xf32>)
  // CHECK:      ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK:        %[[COS:.*]] = math.cos %[[ARG2]]
  // CHECK:        linalg.yield %[[COS]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%arg : tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%arg_scalar: f32, %_: f32):
    %cos = math.cos %arg_scalar : f32
    linalg.yield %cos : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%point]
      : tensor<32x32xf32>[!gml_st.tile<1x1>]
  return { op_label = "consumer" } %result : tensor<1x1xf32>
}

// -----

#transposed = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @transpose_point
// CHECK-SAME:  %[[ARG0:.*]]: tensor<1x2x3x?xf32>, %[[ARG1:.*]]: !gml_st.tile<1x1x1x1>
func.func @transpose_point(%arg: tensor<1x2x3x?xf32>,
    %point: !gml_st.tile<1x1x1x1>) -> tensor<1x1x1x1xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C2:.*]] = arith.constant 2
  // CHECK:      %[[C3:.*]] = arith.constant 3
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C3]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [2, 1, %[[DIM]], 3]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG1]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG1]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG1]][%[[C1]]]
  // CHECK:      %[[OFFSET_1:.*]] = gml_st.offset %[[ARG1]][%[[C2]]]
  // CHECK:      %[[SIZE_1:.*]] = gml_st.size %[[ARG1]][%[[C2]]]
  // CHECK:      %[[OFFSET_2:.*]] = gml_st.offset %[[ARG1]][%[[C3]]]
  // CHECK:      %[[SIZE_2:.*]] = gml_st.size %[[ARG1]][%[[C3]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [1, 2, 3, %[[DIM]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET_0]], %[[OFFSET]], %[[OFFSET_2]], %[[OFFSET_1]]]
  // CHECK-SAME:     [%[[SIZE_0]], %[[SIZE]], %[[SIZE_2]], %[[SIZE_1]]]
  // CHECK-SAME:     [1, 1, 1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [2, 1, %[[DIM]], 3]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]], %[[OFFSET_1]], %[[OFFSET_2]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]]
  // CHECK-SAME:     [1, 1, 1, 1]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[INIT]][%[[TILE_0]]]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_0]]
  // CHECK:      %[[CAST_0:.*]] = tensor.cast %[[MATERIALIZE]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK-SAME:     ins(%[[CAST_0]] : tensor<1x1x1x1xf32>)
  // CHECK-SAME:     outs(%[[CAST]] : tensor<1x1x1x1xf32>)
  // CHECK:      ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK:        linalg.yield %[[ARG2]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %c3 = arith.constant 3 : index
  %d3 = tensor.dim %arg, %c3 : tensor<1x2x3x?xf32>
  %init = linalg.init_tensor [2, 1, %d3, 3] : tensor<2x1x?x3xf32>
  %transpose = linalg.generic {
      indexing_maps = [#transposed, #id],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"],
      op_label = "producer" }
      ins(%arg : tensor<1x2x3x?xf32>) outs(%init : tensor<2x1x?x3xf32>) {
  ^bb0(%a: f32, %_: f32):
    linalg.yield %a : f32
  } -> tensor<2x1x?x3xf32>
  %transpose_sub = gml_st.materialize %transpose[%point]
      : tensor<2x1x?x3xf32>[!gml_st.tile<1x1x1x1>]
  return { op_label = "consumer" } %transpose_sub : tensor<1x1x1x1xf32>
}

// -----

#transposed = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @transpose_tile
// CHECK-SAME:  %[[ARG0:.*]]: tensor<1x2x3x?xf32>, %[[ARG1:.*]]: !gml_st.tile<?x?x?x?>
func.func @transpose_tile(%arg: tensor<1x2x3x?xf32>,
    %tile: !gml_st.tile<?x?x?x?>) -> tensor<?x?x?x?xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C2:.*]] = arith.constant 2
  // CHECK:      %[[C3:.*]] = arith.constant 3
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C3]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [2, 1, %[[DIM]], 3]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG1]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG1]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG1]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG1]][%[[C1]]]
  // CHECK:      %[[OFFSET_1:.*]] = gml_st.offset %[[ARG1]][%[[C2]]]
  // CHECK:      %[[SIZE_1:.*]] = gml_st.size %[[ARG1]][%[[C2]]]
  // CHECK:      %[[OFFSET_2:.*]] = gml_st.offset %[[ARG1]][%[[C3]]]
  // CHECK:      %[[SIZE_2:.*]] = gml_st.size %[[ARG1]][%[[C3]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [1, 2, 3, %[[DIM]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET_0]], %[[OFFSET]], %[[OFFSET_2]], %[[OFFSET_1]]]
  // CHECK-SAME:     [%[[SIZE_0]], %[[SIZE]], %[[SIZE_2]], %[[SIZE_1]]]
  // CHECK-SAME:     [1, 1, 1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:      %[[SPACE_0:.*]] = gml_st.space [2, 1, %[[DIM]], 3]
  // CHECK:      %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]], %[[OFFSET_1]], %[[OFFSET_2]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]]
  // CHECK-SAME:     [1, 1, 1, 1]
  // CHECK:      %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[INIT]][%[[TILE_0]]]
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:     ins(%[[MATERIALIZE]] : tensor<?x?x?x?xf32>)
  // CHECK-SAME:     outs(%[[MATERIALIZE_0]] : tensor<?x?x?x?xf32>)
  // CHECK:      ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK:        linalg.yield %[[ARG2]]
  // CHECK:      return {op_label = "consumer"} %[[GENERIC]]
  %c3 = arith.constant 3 : index
  %d3 = tensor.dim %arg, %c3 : tensor<1x2x3x?xf32>
  %init = linalg.init_tensor [2, 1, %d3, 3] : tensor<2x1x?x3xf32>
  %transposed = linalg.generic {
      indexing_maps = [#transposed, #id],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"],
      op_label = "producer" }
      ins(%arg : tensor<1x2x3x?xf32>) outs(%init : tensor<2x1x?x3xf32>) {
  ^bb0(%a: f32, %_: f32):
    linalg.yield %a : f32
  } -> tensor<2x1x?x3xf32>
  %transposed_sub = gml_st.materialize %transposed[%tile]
      : tensor<2x1x?x3xf32>[!gml_st.tile<?x?x?x?>]
  return { op_label = "consumer" } %transposed_sub : tensor<?x?x?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @empty
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: !gml_st.tile<1x1>
func.func @empty(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                 %pt: !gml_st.tile<1x1>)-> tensor<1x1xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[OFFSET:.*]] = gml_st.offset %[[ARG2]][%[[C0]]]
  // CHECK:      %[[SIZE:.*]] = gml_st.size %[[ARG2]][%[[C0]]]
  // CHECK:      %[[OFFSET_0:.*]] = gml_st.offset %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SIZE_0:.*]] = gml_st.size %[[ARG2]][%[[C1]]]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
  // CHECK:      %[[TILE:.*]] = gml_st.tile %[[SPACE]]
  // CHECK-SAME:     [%[[OFFSET]], %[[OFFSET_0]]]
  // CHECK-SAME:     [%[[SIZE]], %[[SIZE_0]]]
  // CHECK-SAME:     [1, 1]
  // CHECK:      %[[MATERIALIZE:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[CAST:.*]] = tensor.cast %[[MATERIALIZE]]
  // CHECK:      return {op_label = "consumer"} %[[CAST]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%_0: f32, %_1: f32, %arg2: f32):
    linalg.yield %arg2 : f32
  } -> tensor<?x?xf32>
  %elem =  gml_st.materialize %result[%pt] : tensor<?x?xf32>[!gml_st.tile<1x1>]
  return { op_label = "consumer" } %elem : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: @dim_reification_fission
// CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>
func.func @dim_reification_fission(%arg0: tensor<?xf32>) -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: return %[[DIM]]
  %c0 = arith.constant 0 : index
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = tensor.extract %0[%c0] : tensor<1xindex>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_materialize
// CHECK-SAME:  %{{.*}}: tensor<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @dim_reification_materialize(%arg: tensor<?x?xf32>,
    %tile: !gml_st.tile<?x?>) -> index {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[RES:.*]] = gml_st.size %[[TILE]][%[[C0]]]
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %0 = gml_st.materialize %arg[%tile] : tensor<?x?xf32>[!gml_st.tile<?x?>]
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_generic
// CHECK-SAME:  %{{.*}}: tensor<?x?xf32>, %[[INIT:.*]]: tensor<?x?xf32>, %[[IDX:.*]]: index
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @dim_reification_generic(%arg: tensor<?x?xf32>,
    %init: tensor<?x?xf32>, %idx: index) -> index {
  // CHECK-DAG: %[[RES:.*]] = tensor.dim %[[INIT]], %[[IDX]]
  // CHECK:     return %[[RES]]
  %0 = linalg.generic
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%arg : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = math.log %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.dim %0, %idx : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_init_tensor
// CHECK-SAME:  %{{.*}}: index, %[[J:.*]]: index
func.func @dim_reification_init_tensor(%i: index, %j: index) -> index {
  // CHECK: return %[[J]]
  %c1 = arith.constant 1 : index
  %0 = linalg.init_tensor [%i, %j] : tensor<?x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_dynamic_broadcast_in_dim
// CHECK-SAME:  %{{.*}}: tensor<?xf32>, %[[INIT:.*]]: tensor<?x?xf32>
func.func @dim_reification_dynamic_broadcast_in_dim(%arg: tensor<?xf32>,
    %init: tensor<?x?xf32>) -> index {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[RES:.*]] = tensor.dim %[[INIT]], %[[C1]]
  // CHECK:     return %[[RES]] : index
  %c1 = arith.constant 1 : index
  %0 = thlo.dynamic_broadcast_in_dim
      ins(%arg : tensor<?xf32>) outs(%init : tensor<?x?xf32>)
      broadcast_dimensions = [1]
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_concatenate
// CHECK-SAME:  %[[INIT:.*]]: tensor<?x?xi32>, %[[A:.*]]: tensor<?x?xi32>, %[[B:.*]]: tensor<?x?xi32>, %[[C:.*]]: tensor<?x?xi32>
func.func @dim_reification_concatenate(%init : tensor<?x?xi32>,
    %a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> index {
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[INIT]], %[[C1]] : tensor<?x?xi32>
  // CHECK:      return %[[DIM]] : index
  %c1 = arith.constant 1 : index
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>)
      {dimension = 1 : i64}
  %dim = tensor.dim %concat, %c1 : tensor<?x?xi32>
  func.return %dim : index
}
