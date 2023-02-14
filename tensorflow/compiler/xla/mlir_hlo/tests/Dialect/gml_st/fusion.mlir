// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-fusion="producer-label=producer consumer-label=consumer" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>, %i: index, %j: index, %k : index,
    %arg_dim: index) -> tensor<3x4x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %bcast = thlo.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      broadcast_dimensions = [0, 2]
      { op_label = "producer" }

  %bcast_sub = tensor.extract_slice %bcast[%i, %j, %k] [3, 4, %arg_dim] [1, 1, 1]
      : tensor<?x?x?xf32> to tensor<3x4x?xf32>
  func.return { op_label = "consumer" } %bcast_sub : tensor<3x4x?xf32>
}
// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>,
// CHECK-SAME:  %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[ARG_DIM:.*]]: index)

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[C2:.*]] = arith.constant 2
// CHECK-DAG:  %[[C3:.*]] = arith.constant 3
// CHECK:      %[[EXTRACT_0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK:      %[[EXTRACT_1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK:      %[[EXTRACT_2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK:      %[[INIT:.*]] = tensor.empty(%[[EXTRACT_0]], %[[EXTRACT_1]], %[[EXTRACT_2]])
// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// CHECK:      %[[CMPI_0:.*]] = arith.cmpi ne, %[[DIM_0]], %[[EXTRACT_0]]
// CHECK:      %[[CMPI_1:.*]] = arith.cmpi ne, %[[DIM_1]], %[[EXTRACT_2]]
// CHECK:      %[[SELECT:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[I]]
// CHECK:      %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[C0]], %[[K]]
// CHECK:      %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[C3]]
// CHECK:      %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[C1]], %[[ARG_DIM]]
// CHECK:      %[[INIT_SUB:.*]] = tensor.extract_slice %[[INIT]]
// CHECK-SAME:     [%[[I]], %[[J]], %[[K]]] [3, 4, %[[ARG_DIM]]]
// CHECK:      %[[ARG_SUB:.*]] = tensor.extract_slice %[[ARG]]
// CHECK-SAME:     [%[[SELECT]], %[[SELECT_0]]]
// CHECK-SAME:     [%[[SELECT_1]], %[[SELECT_2]]]
// CHECK-SAME:     [1, 1]
// CHECK:      %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
// CHECK-SAME:     ins(%[[ARG_SUB]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<3x4x?xf32>)
// CHECK-SAME:     broadcast_dimensions = [0, 2]
// CHECK:      return {op_label = "consumer"} %[[DYNAMIC]]

// -----

// CHECK-LABEL: @nary_concatenate_with_unit_dims
// CHECK-SAME:  %[[INIT:.*]]: tensor<?x?xi32>, %[[ARG_A:.*]]: tensor<?x?xi32>, %[[ARG_B:.*]]: tensor<?x?xi32>, %[[ARG_C:.*]]: tensor<?x?xi32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[N:.*]]: index
func.func @nary_concatenate_with_unit_dims(%init : tensor<?x?xi32>,
    %a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>, %i: index,
    %j: index, %n: index) -> tensor<?x1xi32> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK:     %[[DIM:.*]] = tensor.dim %[[ARG_A]], %[[C1]]
  // CHECK:     %[[CMPI:.*]] = arith.cmpi ult, %[[J]], %[[DIM]]
  // CHECK:     %[[IF:.*]] = scf.if %[[CMPI]] -> (tensor<?x1xi32>)
  // CHECK:       %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG_A]][%[[I]], %[[J]]] [%[[N]], 1] [1, 1]
  // CHECK:       scf.yield %[[MATERIALIZE]]
  // CHECK:     else
  // CHECK:       %[[SUBI:.*]] = arith.subi %[[J]], %[[DIM]]
  // CHECK:       %[[DIM_0:.*]] = tensor.dim %[[ARG_B]], %[[C1]]
  // CHECK:       %[[CMPI_0:.*]] = arith.cmpi ult, %[[SUBI]], %[[DIM_0]]
  // CHECK:       %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<?x1xi32>)
  // CHECK:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG_B]][%[[I]], %[[SUBI]]] [%[[N]], 1] [1, 1]
  // CHECK:         scf.yield %[[MATERIALIZE_0]]
  // CHECK:       else
  // CHECK:         %[[SUBI_0:.*]] = arith.subi %[[SUBI]], %[[DIM_0]]
  // CHECK:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[ARG_C]][%[[I]], %[[SUBI_0]]] [%[[N]], 1] [1, 1]
  // CHECK:         scf.yield %[[MATERIALIZE_1]]
  // CHECK:       scf.yield %[[IF_0]]
  // CHECK:     return {op_label = "consumer"} %[[IF]]
  %concat = thlo.concatenate ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>,
      %c : tensor<?x?xi32>) outs(%init : tensor<?x?xi32>) dimension = 1
      { op_label = "producer" }
  %tiled_concat = tensor.extract_slice %concat[%i, %j] [%n, 1] [1, 1]
      : tensor<?x?xi32> to tensor<?x1xi32>
  func.return { op_label = "consumer" } %tiled_concat : tensor<?x1xi32>
}

// -----

// CHECK-LABEL: @binary_concatenate_with_unit_dims
// CHECK-SAME:  %[[INIT:.*]]: tensor<?x?xi32>, %[[ARG_A:.*]]: tensor<?x?xi32>, %[[ARG_B:.*]]: tensor<?x?xi32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[N:.*]]: index
func.func @binary_concatenate_with_unit_dims(%init : tensor<?x?xi32>,
    %a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %i: index, %j: index, %n: index)
    -> tensor<?x1xi32> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK:     %[[DIM:.*]] = tensor.dim %[[ARG_A]], %[[C1]]
  // CHECK:     %[[CMPI:.*]] = arith.cmpi ult, %[[J]], %[[DIM]]
  // CHECK:     %[[IF:.*]] = scf.if %[[CMPI]] -> (tensor<?x1xi32>)
  // CHECK:       %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG_A]][%[[I]], %[[J]]] [%[[N]], 1] [1, 1]
  // CHECK:       scf.yield %[[MATERIALIZE]]
  // CHECK:     else
  // CHECK:       %[[SUBI:.*]] = arith.subi %[[J]], %[[DIM]]
  // CHECK:       %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG_B]][%[[I]], %[[SUBI]]] [%[[N]], 1] [1, 1]
  // CHECK:       scf.yield %[[MATERIALIZE_0]]
  // CHECK:     return {op_label = "consumer"} %[[IF]]
  %concat = thlo.concatenate ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>) dimension = 1 { op_label = "producer" }
  %tiled_concat = tensor.extract_slice %concat[%i, %j] [%n, 1] [1, 1]
      : tensor<?x?xi32> to tensor<?x1xi32>
  func.return { op_label = "consumer" } %tiled_concat : tensor<?x1xi32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>, %i: index,
    %j: index, %arg_dim0: index, %arg_dim1: index) -> tensor<?x?xf32> {
  %init = tensor.empty() : tensor<32x32xf32>
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
  %result = tensor.extract_slice %linalg[%i, %j] [%arg_dim0, %arg_dim1] [1, 1]
      : tensor<32x32xf32> to tensor<?x?xf32>
  return { op_label = "consumer" } %result : tensor<?x?xf32>
}
// CHECK-LABEL: @add
// CHECK-SAME:  %[[LHS:[a-z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:  %[[RHS:[a-z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:  %[[I:[a-z0-9]+]]: index, %[[J:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[ARG_DIM0:[a-z0-9]+]]: index, %[[ARG_DIM1:[a-z0-9]+]]: index)

// CHECK:      %[[INIT:.*]] = tensor.empty()
// CHECK:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS]]
// CHECK-SAME:     [%[[I]], %[[J]]] [%[[ARG_DIM0]], %[[ARG_DIM1]]]
// CHECK:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS]]
// CHECK-SAME:     [%[[I]], %[[J]]] [%[[ARG_DIM0]], %[[ARG_DIM1]]]
// CHECK:      %[[INIT_SUB:.*]] = tensor.extract_slice %[[INIT]]
// CHECK-SAME:     [%[[I]], %[[J]]] [%[[ARG_DIM0]], %[[ARG_DIM1]]]
// CHECK:      %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>,
// CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// CHECK:      ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32):
// CHECK:        %[[ADDF:.*]] = arith.addf %[[ARG3]], %[[ARG4]]
// CHECK:        linalg.yield %[[ADDF]]
// CHECK:      return {op_label = "consumer"} %[[GENERIC]]

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @empty(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
    %i: index, %j: index) -> tensor<1x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "producer" }
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%_0: f32, %_1: f32, %arg2: f32):
    linalg.yield %arg2 : f32
  } -> tensor<?x?xf32>

  %elem =  tensor.extract_slice %result[%i, %j] [1, 1] [1, 1]
    : tensor<?x?xf32> to tensor<1x1xf32>
  return { op_label = "consumer" } %elem : tensor<1x1xf32>
}
// CHECK-LABEL: @empty
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[I:.*]]: index, %[[J:.*]]: index)

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK:      %[[INIT:.*]] = tensor.empty(%[[DIM]], %[[DIM_0]])

// CHECK:      %[[MATERIALIZE:.*]] = tensor.extract_slice %[[INIT]]
// CHECK-SAME:   [%[[I]], %[[J]]] [1, 1]
// CHECK:      return {op_label = "consumer"} %[[MATERIALIZE]]

// -----

func.func @dim_reification_fission(%arg: tensor<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = shape.shape_of %arg : tensor<?xf32> -> tensor<1xindex>
  %1 = tensor.extract %0[%c0] : tensor<1xindex>
  return %1 : index
}
// CHECK-LABEL: @dim_reification_fission
// CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>)

// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: %[[DIM:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK: return %[[DIM]]

// -----

func.func @dim_reification_materialize(%arg: tensor<?x?xf32>,
    %arg_dim0: index, %arg_dim1: index) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %arg[0, 0] [%arg_dim0, %arg_dim1] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  return %1 : index
}
// CHECK-LABEL: @dim_reification_materialize
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[ARG_DIM0:.*]]: index, %[[ARG_DIM1:.*]]: index)
// CHECK:     return %[[ARG_DIM0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @dim_reification_generic(%arg: tensor<?x?xf32>,
    %init: tensor<?x?xf32>, %idx: index) -> index {
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
// CHECK-LABEL: @dim_reification_generic
// CHECK-SAME:  %{{.*}}: tensor<?x?xf32>, %[[INIT:[a-z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[IDX:[a-z0-9]+]]: index

// CHECK: %[[RES:.*]] = tensor.dim %[[INIT]], %[[IDX]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: @dim_reification_init_tensor
// CHECK-SAME:  %{{.*}}: index, %[[J:.*]]: index
func.func @dim_reification_init_tensor(%i: index, %j: index) -> index {
  // CHECK: return %[[J]]
  %c1 = arith.constant 1 : index
  %0 = tensor.empty(%i, %j) : tensor<?x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}

// -----

func.func @dim_reification_dynamic_broadcast_in_dim(%arg: tensor<?xf32>,
    %init: tensor<?x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %0 = thlo.dynamic_broadcast_in_dim
      ins(%arg : tensor<?xf32>) outs(%init : tensor<?x?xf32>)
      broadcast_dimensions = [1]
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}
// CHECK-LABEL: @dim_reification_dynamic_broadcast_in_dim
// CHECK-SAME:  %{{.*}}: tensor<?xf32>, %[[INIT:.*]]: tensor<?x?xf32>

// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[RES:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK:     return %[[RES]] : index

// -----

func.func @dim_reification_concatenate(%init : tensor<?x?xi32>,
    %a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> index {
  %c1 = arith.constant 1 : index
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>)
      dimension = 1
  %dim = tensor.dim %concat, %c1 : tensor<?x?xi32>
  func.return %dim : index
}
// CHECK-LABEL: @dim_reification_concatenate
// CHECK-SAME:  %[[INIT:.*]]: tensor<?x?xi32>, %[[A:.*]]: tensor<?x?xi32>, %[[B:.*]]: tensor<?x?xi32>, %[[C:.*]]: tensor<?x?xi32>

// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[INIT]], %[[C1]] : tensor<?x?xi32>
// CHECK:      return %[[DIM]] : index

// -----

#map = affine_map<(d0) -> (d0)>

func.func @fusion_into_materialize_element(
    %input : tensor<?xf32>, %init : tensor<?xf32>, %idx : index) -> f32 {
  %neg = linalg.generic
      {indexing_maps = [#map, #map], iterator_types = ["parallel"],
       op_label="producer" }
      ins(%input : tensor<?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.negf %in : f32
    linalg.yield %0 : f32
  } -> tensor<?xf32>
  %res_slice = tensor.extract_slice %neg[%idx] [1] [1]
    : tensor<?xf32> to tensor<1xf32>
  %c0 = arith.constant 0 : index
  %res = tensor.extract %res_slice[%c0] { op_label="consumer" } : tensor<1xf32>
  return  %res : f32
}
// CHECK-LABEL: @fusion_into_materialize_element
// CHECK-COUNT-2: tensor.extract_slice
// CHECK:         linalg.generic

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%lhs: tensor<128x16xf32>,
                  %rhs: tensor<16x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill { op_label = "producer" } ins(%cst : f32)
      outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
  %matmul = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c128, %c256)
      step (%c8, %c8) outs (%out = %fill: tensor<128x256xf32>) {
    %lhs_sub = tensor.extract_slice %lhs[%i, 0] [8, 16] [1, 1]
      : tensor<128x16xf32> to tensor<8x16xf32>
    %rhs_sub = tensor.extract_slice %rhs[0, %j] [16, 8] [1, 1]
      : tensor<16x256xf32> to tensor<16x8xf32>
    %out_sub = tensor.extract_slice %out[%i, %j] [8, 8] [1, 1]
      : tensor<128x256xf32> to tensor<8x8xf32>

    %matmul_sub = linalg.matmul { op_label="consumer" }
      ins(%lhs_sub, %rhs_sub : tensor<8x16xf32>, tensor<16x8xf32>)
      outs(%out_sub : tensor<8x8xf32>) -> tensor<8x8xf32>

    %out_tile = gml_st.tile [%i, %j] [8, 8] [1, 1] : !gml_st.tile<8x8>
    gml_st.set_yield %matmul_sub into %out[%out_tile]
      : tensor<8x8xf32> into tensor<128x256xf32>[!gml_st.tile<8x8>]
  } : tensor<128x256xf32>
  return %matmul : tensor<128x256xf32>
}
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %[[LHS:.*]]: tensor<128x16xf32>,
// CHECK-SAME:    %[[RHS:.*]]: tensor<16x256xf32>) -> tensor<128x256xf32> {
// CHECK:      %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<128x256xf32>
// CHECK:       gml_st.parallel (%[[I:[a-z0-9]+]], %[[J:[a-z0-9]+]])
// CHECK-SAME:     outs (%[[OUT_:.*]] = %[[EMPTY]]:
// CHECK:        %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]] [8, 8] [1, 1]
// CHECK:        %[[FILL:.*]] = linalg.fill
// CHECK-SAME:     outs(%[[OUT_SUB]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:        %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:     outs(%[[FILL]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:        %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]] [8, 8] [1, 1]
// CHECK:        gml_st.set_yield %[[MATMUL]] into %[[OUT_]][%[[OUT_TILE]]]
// CHECK-SAME:     : tensor<8x8xf32> into tensor<128x256xf32>[!gml_st.tile<8x8>]
