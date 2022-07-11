// RUN: mlir-hlo-opt %s --gml-tiling="tile-sizes=256,512" | \
// RUN: FileCheck %s --check-prefix=CHECK-TILE

// RUN: mlir-hlo-opt %s --gml-tiling="tile-sizes=1,1" | \
// RUN: FileCheck %s --check-prefix=CHECK-POINT

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
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}


// CHECK-TILE:      @add
// CHECK-TILE-SAME: %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-TILE-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-TILE-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-TILE-DAG:  %[[C256:.*]] = arith.constant 256
// CHECK-TILE-DAG:  %[[C512:.*]] = arith.constant 512
// CHECK-TILE:      %[[GENERIC:.*]] = linalg.generic
// CHECK-TILE-DAG:  %[[D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-TILE-DAG:  %[[D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-TILE-DAG:  %[[SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// CHECK-TILE-DAG:  %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
// CHECK-TILE-DAG:  %[[D0_:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-TILE-DAG:  %[[D1_:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-TILE:      %[[RES:.*]] = gml_st.parallel
// CHECK-TILE-SAME:     (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-TILE-SAME:     to (%[[D0_]], %[[D1_]])
// CHECK-TILE-SAME:     step (%[[C256]], %[[C512]])
// CHECK-TILE-DAG:    %[[I_PLUS_256:.*]] = arith.addi %[[I]], %[[C256]]
// CHECK-TILE-DAG:    %[[IS_PARTIAL0:.*]] = arith.cmpi sgt, %[[I_PLUS_256]], %[[D0_]]
// CHECK-TILE-DAG:    %[[PARTIAL_SIZE0:.*]] = arith.subi %[[D0_]], %[[I]]
// CHECK-TILE-DAG:    %[[SIZE0:.*]] = arith.select %[[IS_PARTIAL0]], %[[PARTIAL_SIZE0]], %[[C256]]
// CHECK-TILE-DAG:    %[[J_PLUS_512:.*]] = arith.addi %[[J]], %[[C512]]
// CHECK-TILE-DAG:    %[[IS_PARTIAL1:.*]] = arith.cmpi sgt, %[[J_PLUS_512]], %[[D1_]]
// CHECK-TILE-DAG:    %[[PARTIAL_SIZE1:.*]] = arith.subi %[[D1_]], %[[J]]
// CHECK-TILE-DAG:    %[[SIZE1:.*]] = arith.select %[[IS_PARTIAL1]], %[[PARTIAL_SIZE1]], %[[C512]]
// CHECK-TILE-DAG:    %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[SIZE0]], %[[SIZE1]]] [1, 1]
// CHECK-TILE-DAG:    %[[INNER_RES:.*]] = gml_st.materialize %[[GENERIC]][%[[TILE]]]
// CHECK-TILE:        gml_st.set_yield %[[INNER_RES]] into %[[INIT]][%[[TILE]]]
// CHECK-TILE:      return %[[RES]]


// CHECK-POINT:      @add
// CHECK-SAME:       %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-POINT-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-POINT-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-POINT:      %[[GENERIC:.*]] = linalg.generic
// CHECK-POINT-DAG:  %[[D0:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-POINT-DAG:  %[[D1:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-POINT-DAG:  %[[SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// CHECK-POINT-DAG:  %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
// CHECK-POINT-DAG:  %[[D0_:.*]] = tensor.dim %[[GENERIC]], %[[C0]]
// CHECK-POINT-DAG:  %[[D1_:.*]] = tensor.dim %[[GENERIC]], %[[C1]]
// CHECK-POINT:      %[[RES:.*]] = gml_st.parallel
// CHECK-POINT-SAME:     (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-POINT-SAME:     to (%[[D0_]], %[[D1_]])
// CHECK-POINT-SAME:     step (%[[C1]], %[[C1]])
// CHECK-POINT-DAG:    %[[POINT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J]]]
// CHECK-POINT-DAG:    %[[INNER_RES:.*]] = gml_st.materialize %[[GENERIC]][%[[POINT]]]
// CHECK-POINT:        gml_st.set_yield %[[INNER_RES]] into %[[INIT]][%[[POINT]]]
// CHECK-POINT:      return %[[RES]]
