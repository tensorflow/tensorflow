// RUN: mlir-hlo-opt %s --gml-tiling="tile-sizes=1,1" -cse -split-input-file |\
// RUN: FileCheck %s -check-prefix=POINT

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

// POINT-SAME: %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// POINT:        %[[SPACE:.*]] = gml_st.space {{.*}} : !gml_st.tile<?x?>
// POINT:        %{{.*}} = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// POINT-SAME:   outs (%[[OUT_:.*]] = %{{.*}} at %[[SPACE]]
// POINT-SAME:         tensor<?x?xf32> at !gml_st.tile<?x?>) {

// POINT:        %[[PT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J]]]
// POINT-SAME:     !gml_st.tile<?x?> to !gml_st.point
// POINT-NEXT:   %[[LHS_PT:.*]] = gml_st.materialize %[[LHS]] at %[[PT]]
// POINT-SAME:     : tensor<?x?xf32> at !gml_st.point
// POINT-NEXT:   %[[RHS_PT:.*]] = gml_st.materialize %[[RHS]] at %[[PT]]
// POINT-SAME:     : tensor<?x?xf32> at !gml_st.point

// POINT-NEXT:   %[[ADD:.*]] = arith.addf %[[LHS_PT]], %[[RHS_PT]] : f32
// POINT-NEXT:   gml_st.subset_yield %[[ADD]] at %[[PT]] : f32 at !gml_st.point
