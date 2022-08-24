// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tile-to-for="tile-sizes=256,512 tiling-target="op_2d"" \
// RUN: | FileCheck %s

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
// CHECK-LABEL: func.func @add
// CHECK-SAME:    (%[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index

// CHECK:     %[[INIT:.*]] = linalg.init_tensor
// CHECK:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:     %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>

// CHECK:     gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:    to (%[[DIM_0]], %[[DIM_1]])
// CHECK-SAME:    step (%[[C256]], %[[C512]])
// CHECK-SAME:    outs (%[[INIT_:.*]] = %[[INIT]]: tensor<?x?xf32>) {

// CHECK:       %[[SIZE_0:.*]] = affine.min #map0(%[[I]])[%[[C256]], %[[DIM_0]]]
// CHECK:       %[[SIZE_1:.*]] = affine.min #map1(%[[J]])[%[[C512]], %[[DIM_1]]]

// CHECK:       %[[LHS_T:.*]] = gml_st.tile
// CHECK-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// CHECK:       %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_T]]]

// CHECK:       %[[RHS_T:.*]] = gml_st.tile
// CHECK-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// CHECK:       %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_T]]]

// CHECK:       %[[INIT_T:.*]] = gml_st.tile
// CHECK-SAME:    [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// CHECK:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT_]][%[[INIT_T]]]

// CHECK:       %[[SUM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[LHS_SUB]], %[[RHS_SUB]]
// CHECK-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// CHECK:       gml_st.set_yield %[[SUM:.*]] into %[[INIT_]][%[[INIT_T]]

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
// CHECK:   func.func @reduce_row(%[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                    %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:     %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]] : f32)
// CHECK:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]]
// CHECK:     %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]]

// CHECK:     gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:    to (%[[DIM_0]], %[[DIM_1]])
// CHECK-SAME:    step (%[[C256]], %[[C512]])
// CHECK-SAME:    outs (%[[INIT_:.*]] = %[[FILL]]: tensor<?xf32>) {

// CHECK:      %[[SIZE_0:.*]] = affine.min {{.*}}(%[[I]])[%[[C256]], %[[DIM_0]]]
// CHECK:      %[[SIZE_1:.*]] = affine.min {{.*}}(%[[J]])[%[[C512]], %[[DIM_1]]]

// CHECK:      %[[LHS_T:.*]] = gml_st.tile
// CHECK-SAME:   [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// CHECK:      %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_T]]]

// CHECK:      %[[RHS_T:.*]] = gml_st.tile
// CHECK-SAME:   [%[[I]], %[[J]]] [%[[SIZE_0]], %[[SIZE_1]]] [1, 1]
// CHECK:      %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_T]]]

// CHECK:      %[[INIT_T:.*]] = gml_st.tile %{{.*}}[%[[I]]] [%[[SIZE_0]]] [1]
// CHECK:      %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT_]][%[[INIT_T]]]

// CHECK:      %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]]
// CHECK-SAME:   outs(%[[INIT_SUB]] : tensor<?xf32>)
// CHECK:      gml_st.set_yield %[[REDUCE:.*]] into %[[INIT_]][%[[INIT_T]]]

