// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false tiling-target="op_2d"" \
// RUN: | FileCheck %s --check-prefix=FOR

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=true tiling-target="op_2d"" \
// RUN: | FileCheck %s --check-prefix=PARALLEL

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
      op_label = "op_2d"}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  func.return %add : tensor<?x?xf32>
}
// FOR-LABEL: func.func @add
// FOR-SAME:    (%[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// FOR-DAG: %[[C0:.*]] = arith.constant 0 : index
// FOR-DAG: %[[C1:.*]] = arith.constant 1 : index
// FOR-DAG: %[[C256:.*]] = arith.constant 256 : index
// FOR-DAG: %[[C512:.*]] = arith.constant 512 : index

// FOR:     %[[INIT:.*]] = linalg.init_tensor
// FOR:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// FOR:     %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>

// FOR:     gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// FOR-SAME:    to (%[[DIM_0]], %[[DIM_1]])
// FOR-SAME:    step (%[[C256]], %[[C512]])
// FOR-SAME:    outs (%[[INIT_:.*]] = %[[INIT]]: tensor<?x?xf32>) {

// FOR:       %[[SIZE_0:.*]] = affine.min #map0(%[[I]])[%[[C256]], %[[DIM_0]]]
// FOR:       %[[SIZE_1:.*]] = affine.min #map1(%[[J]])[%[[C512]], %[[DIM_1]]]

// FOR:       %[[LHS_T:.*]] = gml_st.tile
// FOR-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// FOR:       %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_T]]]

// FOR:       %[[RHS_T:.*]] = gml_st.tile
// FOR-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// FOR:       %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_T]]]

// FOR:       %[[INIT_T:.*]] = gml_st.tile
// FOR-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// FOR:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT_]][%[[INIT_T]]]

// FOR:       %[[SUM:.*]] = linalg.generic
// FOR-SAME:    ins(%[[LHS_SUB]], %[[RHS_SUB]]
// FOR-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// FOR:       gml_st.set_yield %[[SUM:.*]] into %[[INIT_]][%[[INIT_T]]

// PARALLEL-LABEL: func.func @add
// PARALLEL-SAME:    (%[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// PARALLEL-DAG: %[[C0:.*]] = arith.constant 0 : index
// PARALLEL-DAG: %[[C1:.*]] = arith.constant 1 : index
// PARALLEL-DAG: %[[C256:.*]] = arith.constant 256 : index
// PARALLEL-DAG: %[[C512:.*]] = arith.constant 512 : index

// PARALLEL:     %[[INIT:.*]] = linalg.init_tensor
// PARALLEL:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// PARALLEL:     %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>

// PARALLEL:     gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// PARALLEL-SAME:    to (%[[DIM_0]], %[[DIM_1]])
// PARALLEL-SAME:    step (%[[C256]], %[[C512]]) {

// PARALLEL:       %[[SIZE_0:.*]] = affine.min #map0(%[[I]])[%[[C256]], %[[DIM_0]]]
// PARALLEL:       %[[SIZE_1:.*]] = affine.min #map1(%[[J]])[%[[C512]], %[[DIM_1]]]

// PARALLEL:       %[[LHS_T:.*]] = gml_st.tile
// PARALLEL-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// PARALLEL:       %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_T]]]

// PARALLEL:       %[[RHS_T:.*]] = gml_st.tile
// PARALLEL-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// PARALLEL:       %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_T]]]

// PARALLEL:       %[[INIT_T:.*]] = gml_st.tile
// PARALLEL-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// PARALLEL:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[INIT_T]]]

// PARALLEL:       %[[SUM:.*]] = linalg.generic
// PARALLEL-SAME:    ins(%[[LHS_SUB]], %[[RHS_SUB]]
// PARALLEL-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// PARALLEL:       gml_st.set_yield %[[SUM:.*]] into %[[INIT]][%[[INIT_T]]

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
    op_label = "op_2d"}
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %o : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum_of_prod : tensor<?xf32>
}
// FOR:   func.func @reduce_row(%[[LHS:.*]]: tensor<?x?xf32>,
// FOR-SAME:                    %[[RHS:.*]]: tensor<?x?xf32>)

// FOR-DAG: %[[C0:.*]] = arith.constant 0 : index
// FOR-DAG: %[[C1:.*]] = arith.constant 1 : index
// FOR-DAG: %[[C256:.*]] = arith.constant 256 : index
// FOR-DAG: %[[C512:.*]] = arith.constant 512 : index
// FOR-DAG: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32

// FOR:     %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]] : f32)
// FOR:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]]
// FOR:     %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]]

// FOR:     gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// FOR-SAME:    to (%[[DIM_0]], %[[DIM_1]])
// FOR-SAME:    step (%[[C256]], %[[C512]])
// FOR-SAME:    outs (%[[INIT_:.*]] = %[[FILL]]: tensor<?xf32>) {

// FOR:      %[[SIZE_0:.*]] = affine.min {{.*}}(%[[I]])[%[[C256]], %[[DIM_0]]]
// FOR:      %[[SIZE_1:.*]] = affine.min {{.*}}(%[[J]])[%[[C512]], %[[DIM_1]]]

// FOR:      %[[LHS_T:.*]] = gml_st.tile
// FOR-SAME:   [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// FOR:      %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_T]]]

// FOR:      %[[RHS_T:.*]] = gml_st.tile
// FOR-SAME:   [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// FOR:      %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_T]]]

// FOR:      %[[INIT_T:.*]] = gml_st.tile %{{.*}}[%[[I]]] [%[[SIZE_0]]] [1]
// FOR:      %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT_]][%[[INIT_T]]]

// FOR:      %[[REDUCE:.*]] = linalg.generic
// FOR-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]]
// FOR-SAME:   outs(%[[INIT_SUB]] : tensor<?xf32>)
// FOR:      gml_st.set_yield %[[REDUCE:.*]] into %[[INIT_]][%[[INIT_T]]]

// PARALLEL-NOT: gml_st.parallel
