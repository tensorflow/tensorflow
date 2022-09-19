// RUN: mlir-hlo-opt %s --gml-deprecated-tiling="tile-sizes=[256,512]" | \
// RUN: FileCheck %s --check-prefix=CHECK-TILE

// RUN: mlir-hlo-opt %s --gml-deprecated-tiling="tile-sizes=[1,1]" | \
// RUN: FileCheck %s --check-prefix=CHECK-POINT

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %generic = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %generic_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32    
    linalg.yield %generic_scalar : f32
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

// CHECK-TILE-LABEL: @add
// CHECK-TILE-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>
//                   Original ops remain untouched.
// CHECK-TILE:       %[[GENERIC:.*]] = linalg.generic
// CHECK-TILE-SAME:      ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
//                   Create init tensor for tiled ploop.
// CHECK-TILE-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-TILE-DAG:   %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-TILE-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-TILE-DAG:   %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-TILE:       %[[INIT:.*]] = linalg.init_tensor [%[[GENERIC_D0]], %[[GENERIC_D1]]]
//                   Create root space for tiles.
// CHECK-TILE-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-TILE-DAG:   %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-TILE-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-TILE-DAG:   %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-TILE:       %[[SPACE:.*]] = gml_st.space [%[[GENERIC_D0]], %[[GENERIC_D1]]]
//                   Create lower bounds as 0s.
// CHECK-TILE:       %[[LB_C0:.*]] = arith.constant 0
//                   Create upper bounds.
// CHECK-TILE:       %[[C0:.*]] = arith.constant 0
// CHECK-TILE:       %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-TILE:       %[[C1:.*]] = arith.constant 1
// CHECK-TILE:       %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
//                   Create step constants.
// CHECK-TILE-DAG:   %[[C256:.*]] = arith.constant 256
// CHECK-TILE-DAG:   %[[C512:.*]] = arith.constant 512
//                   Create ploop.
// CHECK-TILE:       %[[TILED_GENERIC:.*]] = gml_st.parallel
// CHECK-TILE-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[LB_C0]], %[[LB_C0]]) 
// CHECK-TILE-SAME:      to (%[[GENERIC_D0]], %[[GENERIC_D1]]) 
// CHECK-TILE-SAME:      step (%[[C256]], %[[C512]])
// CHECK-TILE:         %[[REMAINDER_D0:.*]] = arith.subi %[[GENERIC_D0]], %[[I]]
// CHECK-TILE:         %[[TILE_SIZE_D0:.*]] = arith.minsi %[[C256]], %[[REMAINDER_D0]]
// CHECK-TILE:         %[[REMAINDER_D1:.*]] = arith.subi %[[GENERIC_D1]], %[[J]]
// CHECK-TILE:         %[[TILE_SIZE_D1:.*]] = arith.minsi %[[C512]], %[[REMAINDER_D1]]
// CHECK-TILE:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[TILE_SIZE_D0]], %[[TILE_SIZE_D1]]] [1, 1]
// CHECK-TILE:         %[[INNER_GENERIC:.*]] = gml_st.materialize %[[GENERIC]][%[[TILE]]]
// CHECK-TILE:         gml_st.set_yield %[[INNER_GENERIC]] into %[[INIT]][%[[TILE]]]
// CHECK-TILE:       return %[[TILED_GENERIC]]

// CHECK-POINT-LABEL: @add
// CHECK-POINT-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>
//                    Original ops remain untouched.
// CHECK-POINT:       %[[GENERIC:.*]] = linalg.generic
// CHECK-POINT-SAME:      ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
//                    Create init tensor for tiled ploop.
// CHECK-POINT-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-POINT-DAG:   %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-POINT-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-POINT-DAG:   %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-POINT:       %[[INIT:.*]] = linalg.init_tensor [%[[GENERIC_D0]], %[[GENERIC_D1]]]
//                    Create root space for tiles.
// CHECK-POINT-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-POINT-DAG:   %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-POINT-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-POINT-DAG:   %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-POINT:       %[[SPACE:.*]] = gml_st.space [%[[GENERIC_D0]], %[[GENERIC_D1]]]
//                    Create lower bounds as 0s.
// CHECK-POINT:       %[[LB_C0:.*]] = arith.constant 0
//                    Create upper bounds.
// CHECK-POINT:       %[[C0:.*]] = arith.constant 0
// CHECK-POINT:       %[[GENERIC_D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-POINT:       %[[C1:.*]] = arith.constant 1
// CHECK-POINT:       %[[GENERIC_D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
//                    Create step constants.
// CHECK-POINT:       %[[C1:.*]] = arith.constant 1
// CHECK-POINT:       %[[C1_:.*]] = arith.constant 1
//                    Create ploop.
// CHECK-POINT:       %[[TILED_GENERIC:.*]] = gml_st.parallel
// CHECK-POINT-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[LB_C0]], %[[LB_C0]]) 
// CHECK-POINT-SAME:      to (%[[GENERIC_D0]], %[[GENERIC_D1]]) 
// CHECK-POINT-SAME:      step (%[[C1]], %[[C1_]])
// CHECK-POINT-DAG:     %[[POINT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J]]] 
// CHECK-POINT-DAG:     %[[INNER_GENERIC:.*]] = gml_st.materialize %[[GENERIC]][%[[POINT]]]
// CHECK-POINT:         gml_st.set_yield %[[INNER_GENERIC]] into %[[INIT]][%[[POINT]]]
// CHECK-POINT:       return %[[TILED_GENERIC]]

