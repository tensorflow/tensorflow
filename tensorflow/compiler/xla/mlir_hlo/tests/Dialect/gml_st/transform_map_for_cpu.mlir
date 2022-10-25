// RUN: mlir-hlo-opt %s --gml-st-cpu-transform-map="tile-size=8" | FileCheck %s

func.func @map_unary(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                  -> tensor<?x?xf32> {
  %abs = linalg.map
         ins(%input:tensor<?x?xf32>)
         outs(%init:tensor<?x?xf32>)
         (%input_elem: f32) {
           %0 = math.absf %input_elem: f32
           linalg.yield %0: f32
         }
  func.return %abs : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @map_unary(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<?x?xf32>)

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8

// CHECK-DAG:  %[[DIM_0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK-DAG:  %[[DIM_1:.*]] = tensor.dim %[[INPUT]], %[[C1]]

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]]) to (%[[DIM_0]], %[[DIM_1]])
// CHECK-SAME:     step (%[[C1]], %[[C8]]) {
// CHECK:        %[[MIN_DIM:.*]] = affine.min #map(%[[J]])[%[[DIM_1]]]
// CHECK-NEXT:   %[[INPUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK-SAME:                          [1, %[[MIN_DIM]]] [1, 1]
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = gml_st.materialize %[[INPUT]]
// CHECK-NEXT:   %[[INIT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK-SAME:                          [1, %[[MIN_DIM]]] [1, 1]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:     ins(%[[INPUT_SLICE]] : tensor<1x?xf32>)
// CHECK-SAME:     outs(%[[INIT_SLICE]] : tensor<1x?xf32>)
// CHECK-SAME:     (%[[IN_ELEM:.*]]: f32) {
// CHECK-NEXT:       %[[RES_ELEM:.*]] = math.absf %[[IN_ELEM]] : f32
// CHECK-NEXT:       linalg.yield %[[RES_ELEM]] : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   gml_st.set_yield %[[MAPPED]] into %[[INIT]][%[[INIT_TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]
