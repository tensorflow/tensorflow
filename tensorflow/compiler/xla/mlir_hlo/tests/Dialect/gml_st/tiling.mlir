// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-2d" \
// RUN:     --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-3d" | \
// RUN: FileCheck %s --check-prefix=CHECK-SEQUENTIAL

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling="tile-sizes=256,512 distribute=true op-label=tile-2d" | \
// RUN: FileCheck %s --check-prefix=CHECK-PARALLEL


#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "tile-2d"}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  func.return %add : tensor<?x?xf32>
}


// CHECK-SEQUENTIAL-LABEL: @add
// CHECK-SEQUENTIAL-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>

// CHECK-SEQUENTIAL:       %[[C0:.*]] = arith.constant 0
// CHECK-SEQUENTIAL:       %[[C1:.*]] = arith.constant 1
// CHECK-SEQUENTIAL:       %[[C256:.*]] = arith.constant 256
// CHECK-SEQUENTIAL:       %[[C512:.*]] = arith.constant 512
// CHECK-SEQUENTIAL:       %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-SEQUENTIAL:       %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-SEQUENTIAL:       %[[INIT:.*]] = linalg.init_tensor [%[[DIM]], %[[DIM_0]]]
// CHECK-SEQUENTIAL:       %[[DIM_1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-SEQUENTIAL:       %[[DIM_2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-SEQUENTIAL:       %[[FOR:.*]] = gml_st.for (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SEQUENTIAL-SAME:      to (%[[DIM_1]], %[[DIM_2]])
// CHECK-SEQUENTIAL-SAME:      step (%[[C256]], %[[C512]])
// CHECK-SEQUENTIAL-SAME:      outs (%[[ARG4:.*]] = %[[INIT]]: tensor<?x?xf32>)
// CHECK-SEQUENTIAL:         %[[MIN:.*]] = affine.min #map0(%[[ARG2]])[%[[C256]], %[[DIM_1]]]
// CHECK-SEQUENTIAL:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG3]])[%[[C512]], %[[DIM_2]]]
// CHECK-SEQUENTIAL:         %[[DIM_3:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-SEQUENTIAL:         %[[DIM_4:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-SEQUENTIAL:         %[[SPACE:.*]] = gml_st.space [%[[DIM_3]], %[[DIM_4]]]
// CHECK-SEQUENTIAL:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK-SEQUENTIAL:         %[[DIM_5:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK-SEQUENTIAL:         %[[DIM_6:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK-SEQUENTIAL:         %[[SPACE_0:.*]] = gml_st.space [%[[DIM_5]], %[[DIM_6]]]
// CHECK-SEQUENTIAL:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
// CHECK-SEQUENTIAL:         %[[DIM_7:.*]] = tensor.dim %[[ARG4]], %[[C0]]
// CHECK-SEQUENTIAL:         %[[DIM_8:.*]] = tensor.dim %[[ARG4]], %[[C1]]
// CHECK-SEQUENTIAL:         %[[SPACE_1:.*]] = gml_st.space [%[[DIM_7]], %[[DIM_8]]]
// CHECK-SEQUENTIAL:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG4]][%[[TILE_1]]]
// CHECK-SEQUENTIAL:         %[[GENERIC:.*]] = linalg.generic
// CHECK-SEQUENTIAL-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-SEQUENTIAL-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SEQUENTIAL-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-SEQUENTIAL-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-SEQUENTIAL:         ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
// CHECK-SEQUENTIAL:           %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK-SEQUENTIAL:           linalg.yield %[[ADDF]]
// CHECK-SEQUENTIAL:         gml_st.set_yield %[[GENERIC]] into %[[ARG4]][%[[TILE_1]]]
// CHECK-SEQUENTIAL:       return %[[FOR]]


// CHECK-PARALLEL-LABEL: @add
// CHECK-PARALLEL-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>

// CHECK-PARALLEL:       %[[C0:.*]] = arith.constant 0
// CHECK-PARALLEL:       %[[C1:.*]] = arith.constant 1
// CHECK-PARALLEL:       %[[C256:.*]] = arith.constant 256
// CHECK-PARALLEL:       %[[C512:.*]] = arith.constant 512
// CHECK-PARALLEL:       %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-PARALLEL:       %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-PARALLEL:       %[[INIT:.*]] = linalg.init_tensor [%[[DIM]], %[[DIM_0]]]
// CHECK-PARALLEL:       %[[DIM_1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-PARALLEL:       %[[DIM_2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-PARALLEL:       %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-PARALLEL-SAME:      to (%[[DIM_1]], %[[DIM_2]])
// CHECK-PARALLEL-SAME:      step (%[[C256]], %[[C512]])
// CHECK-PARALLEL:         %[[MIN:.*]] = affine.min #map0(%[[ARG2]])[%[[C256]], %[[DIM_1]]]
// CHECK-PARALLEL:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG3]])[%[[C512]], %[[DIM_2]]]
// CHECK-PARALLEL:         %[[DIM_3:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-PARALLEL:         %[[DIM_4:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-PARALLEL:         %[[SPACE:.*]] = gml_st.space [%[[DIM_3]], %[[DIM_4]]]
// CHECK-PARALLEL:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK-PARALLEL:         %[[DIM_5:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK-PARALLEL:         %[[DIM_6:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK-PARALLEL:         %[[SPACE_0:.*]] = gml_st.space [%[[DIM_5]], %[[DIM_6]]]
// CHECK-PARALLEL:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
// CHECK-PARALLEL:         %[[DIM_7:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-PARALLEL:         %[[DIM_8:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK-PARALLEL:         %[[SPACE_1:.*]] = gml_st.space [%[[DIM_7]], %[[DIM_8]]]
// CHECK-PARALLEL:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE_1]]]
// CHECK-PARALLEL:         %[[GENERIC:.*]] = linalg.generic
// CHECK-PARALLEL-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-PARALLEL-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-PARALLEL:         ^bb0(%[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32):
// CHECK-PARALLEL:           %[[ADDF:.*]] = arith.addf %[[ARG4]], %[[ARG5]]
// CHECK-PARALLEL:           linalg.yield %[[ADDF]]
// CHECK-PARALLEL:         gml_st.set_yield %[[GENERIC]] into %[[INIT]][%[[TILE_1]]]
// CHECK-PARALLEL:       return %[[PARALLEL]]

// -----

func.func @reduce_row(%lhs: tensor<?x?xf32>,
                      %rhs: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0] : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32)
                      outs(%init : tensor<?xf32>) -> tensor<?xf32>
  %sum_of_prod = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"],
    op_label = "tile-2d"}
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %o : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum_of_prod : tensor<?xf32>
}


// CHECK-SEQUENTIAL-LABEL: @reduce_row
// CHECK-SEQUENTIAL-SAME:  %[[ARG0_0:.*]]: tensor<?x?xf32>, %[[ARG1_0:.*]]: tensor<?x?xf32>

// CHECK-SEQUENTIAL:       %[[C0_0:.*]] = arith.constant 0
// CHECK-SEQUENTIAL:       %[[C1_0:.*]] = arith.constant 1
// CHECK-SEQUENTIAL:       %[[C256_0:.*]] = arith.constant 256
// CHECK-SEQUENTIAL:       %[[C512_0:.*]] = arith.constant 512
// CHECK-SEQUENTIAL:       %[[CST:.*]] = arith.constant 0.000000e+00
// CHECK-SEQUENTIAL:       %[[DIM_9:.*]] = tensor.dim %[[ARG0_0]], %[[C0_0]]
// CHECK-SEQUENTIAL:       %[[INIT_0:.*]] = linalg.init_tensor [%[[DIM_9]]]
// CHECK-SEQUENTIAL:       %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT_0]] : tensor<?xf32>)
// CHECK-SEQUENTIAL:       %[[DIM_10:.*]] = tensor.dim %[[ARG0_0]], %[[C0_0]]
// CHECK-SEQUENTIAL:       %[[DIM_11:.*]] = tensor.dim %[[ARG0_0]], %[[C1_0]]
// CHECK-SEQUENTIAL:       %[[FOR_0:.*]] = gml_st.for (%[[ARG2_0:.*]], %[[ARG3_0:.*]]) = (%[[C0_0]], %[[C0_0]])
// CHECK-SEQUENTIAL-SAME:      to (%[[DIM_10]], %[[DIM_11]])
// CHECK-SEQUENTIAL-SAME:      step (%[[C256_0]], %[[C512_0]])
// CHECK-SEQUENTIAL-SAME:      outs (%[[ARG4_0:.*]] = %[[FILL]]: tensor<?xf32>)
// CHECK-SEQUENTIAL:         %[[MIN_1:.*]] = affine.min #map0(%[[ARG2_0]])[%[[C256_0]], %[[DIM_10]]]
// CHECK-SEQUENTIAL:         %[[MIN_2:.*]] = affine.min #map1(%[[ARG3_0]])[%[[C512_0]], %[[DIM_11]]]
// CHECK-SEQUENTIAL:         %[[DIM_12:.*]] = tensor.dim %[[ARG0_0]], %[[C0_0]]
// CHECK-SEQUENTIAL:         %[[DIM_13:.*]] = tensor.dim %[[ARG0_0]], %[[C1_0]]
// CHECK-SEQUENTIAL:         %[[SPACE_2:.*]] = gml_st.space [%[[DIM_12]], %[[DIM_13]]]
// CHECK-SEQUENTIAL:         %[[TILE_2:.*]] = gml_st.tile %[[SPACE_2]] [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[ARG0_0]][%[[TILE_2]]]
// CHECK-SEQUENTIAL:         %[[DIM_14:.*]] = tensor.dim %[[ARG1_0]], %[[C0_0]]
// CHECK-SEQUENTIAL:         %[[DIM_15:.*]] = tensor.dim %[[ARG1_0]], %[[C1_0]]
// CHECK-SEQUENTIAL:         %[[SPACE_3:.*]] = gml_st.space [%[[DIM_14]], %[[DIM_15]]]
// CHECK-SEQUENTIAL:         %[[TILE_3:.*]] = gml_st.tile %[[SPACE_3]] [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[ARG1_0]][%[[TILE_3]]]
// CHECK-SEQUENTIAL:         %[[DIM_16:.*]] = tensor.dim %[[ARG4_0]], %[[C0_0]]
// CHECK-SEQUENTIAL:         %[[SPACE_4:.*]] = gml_st.space [%[[DIM_16]]]
// CHECK-SEQUENTIAL:         %[[TILE_4:.*]] = gml_st.tile %[[SPACE_4]] [%[[ARG2_0]]] [%[[MIN_1]]] [1]
// CHECK-SEQUENTIAL:         %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[ARG4_0]][%[[TILE_4]]]
// CHECK-SEQUENTIAL:         %[[GENERIC_0:.*]] = linalg.generic
// CHECK-SEQUENTIAL-SAME:        iterator_types = ["parallel", "reduction"]}
// CHECK-SEQUENTIAL-SAME:        ins(%[[MATERIALIZE_2]], %[[MATERIALIZE_3]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SEQUENTIAL-SAME:        outs(%[[MATERIALIZE_4]] : tensor<?xf32>)
// CHECK-SEQUENTIAL-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-SEQUENTIAL:         ^bb0(%[[ARG5_0:.*]]: f32, %[[ARG6_0:.*]]: f32, %[[ARG7_0:.*]]: f32):
// CHECK-SEQUENTIAL:           %[[MULF:.*]] = arith.mulf %[[ARG5_0]], %[[ARG6_0]]
// CHECK-SEQUENTIAL:           %[[ADDF_0:.*]] = arith.addf %[[MULF]], %[[ARG7_0]]
// CHECK-SEQUENTIAL:           linalg.yield %[[ADDF_0]]
// CHECK-SEQUENTIAL:         gml_st.set_yield %[[GENERIC_0]] into %[[ARG4_0]][%[[TILE_4]]]
// CHECK-SEQUENTIAL:       return %[[FOR_0]]


// CHECK-PARALLEL-LABEL: @reduce_row
// CHECK-PARALLEL-SAME:  %[[ARG0_0:.*]]: tensor<?x?xf32>, %[[ARG1_0:.*]]: tensor<?x?xf32>

// CHECK-PARALLEL-NOT:   gml_st.parallel
// CHECK-PARALLEL:       %[[RES:.*]] = linalg.generic
// CHECK-PARALLEL-NOT:   gml_st.parallel
// CHECK-PARALLEL:       return %[[RES]]
