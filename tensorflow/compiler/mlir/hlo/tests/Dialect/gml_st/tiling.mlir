// Test tiling to points.
// RUN: mlir-hlo-opt %s --gml-tiling="tile-sizes=1,1" -cse -split-input-file |\
// RUN: FileCheck %s -check-prefix=POINT

// Test tiling to tiles.
// RUN: mlir-hlo-opt %s --gml-tiling="tile-sizes=4,1" -cse -split-input-file |\
// RUN: FileCheck %s -check-prefix=TILE

#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#cwise_2d = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func.func @elemenwise(%lhs : tensor<?x?xf32>,
                      %rhs : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dimX = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dimY = tensor.dim %lhs, %c1 : tensor<?x?xf32>

  %init = linalg.init_tensor [%dimX, %dimY] : tensor<?x?xf32>
  %sum = linalg.generic #cwise_2d
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%l : f32, %r: f32, %o: f32):
      %add = arith.addf %l, %r : f32
      linalg.yield %add : f32
  } -> tensor<?x?xf32>

  func.return %sum : tensor<?x?xf32>
}

// POINT-LABEL: func.func @elemenwise(

// POINT-SAME:   %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// POINT:        %[[SPACE:.*]] = gml_st.space {{.*}} : !gml_st.tile<?x?>
// POINT:        %{{.*}} = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// POINT-SAME:   outs (%{{.*}} at %[[SPACE]]
// POINT-SAME:         tensor<?x?xf32> at !gml_st.tile<?x?>) {

// POINT:        %[[PT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J]]]
// POINT-SAME:     !gml_st.tile<?x?> to !gml_st.point
// POINT-NEXT:   %[[LHS_PT:.*]] = gml_st.materialize %[[LHS]] at %[[PT]]
// POINT-SAME:     : tensor<?x?xf32> at !gml_st.point
// POINT-NEXT:   %[[RHS_PT:.*]] = gml_st.materialize %[[RHS]] at %[[PT]]
// POINT-SAME:     : tensor<?x?xf32> at !gml_st.point

// POINT-NEXT:   %[[ADD:.*]] = arith.addf %[[LHS_PT]], %[[RHS_PT]] : f32
// POINT-NEXT:   gml_st.subset_yield %[[ADD]] at %[[PT]] : f32 at !gml_st.point

// TILE-LABEL: func.func @elemenwise
// TILE-SAME:   %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// TILE-DAG:    %[[C0:.*]] = arith.constant 0 : index
// TILE-DAG:    %[[C1:.*]] = arith.constant 1 : index
// TILE-DAG:    %[[C4:.*]] = arith.constant 4 : index
// TILE: %[[INIT:.*]] = linalg.init_tensor
// TILE: %[[DIM0:.*]] = tensor.dim %[[INIT]], %[[C0]] : tensor<?x?xf32>
// TILE: %[[DIM1:.*]] = tensor.dim %[[INIT]], %[[C1]] : tensor<?x?xf32>
// TILE: %[[SPACE:.*]] = gml_st.space {{.*}} : !gml_st.tile<?x?>
// TILE: %{{.*}} = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// TILE:     (%[[C0]], %[[C0]]) to (%[[DIM0]], %[[DIM1]])
// TILE:     step (%[[C4]], %[[C1]]) outs (%{{.*}} at %[[SPACE]]
// TILE:     tensor<?x?xf32> at !gml_st.tile<?x?>) {

// TILE:  %[[NEXT_IV:.*]] = arith.addi %[[I]], %[[C4]] : index
// TILE:  %[[IS_PARTIAL:.*]] = arith.cmpi sgt, %[[NEXT_IV]], %[[DIM0]] : index
// TILE:  %[[REM:.*]] = arith.subi %[[DIM0]], %[[I]] : index
// TILE:  %[[SIZE:.*]] = arith.select %[[IS_PARTIAL]], %[[REM]], %[[C4]] : index

// TILE:  %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[SIZE]], 1]
// TILE:      [1, 1] : !gml_st.tile<?x?> to !gml_st.tile<?x1>
// TILE:  %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]] at %[[TILE]]
// TILE:  %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]] at %[[TILE]]
// TILE:  %[[OUT_SUB:.*]] = gml_st.materialize %[[INIT]] at %[[TILE]]
// TILE:  %[[LINALG_OP:.*]] = linalg.generic
// TILE:      ins(%[[LHS_SUB]], %[[RHS_SUB]] :
// TILE:      tensor<?x1xf32>, tensor<?x1xf32>)
// TILE:      outs(%[[OUT_SUB]] : tensor<?x1xf32>) {
// TILE:   %[[ADDF:.*]] = arith.addf
// TILE:   linalg.yield %[[ADDF]] : f32
// TILE:  gml_st.subset_yield %[[LINALG_OP]] at %[[TILE]] :
// TILE       tensor<?x1xf32> at !gml_st.tile<?x1>
