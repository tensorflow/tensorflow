// RUN: mlir-hlo-opt %s --gml-st-cpu-transform-map="tile-size=8" \ 
// RUN: --split-input-file \
// RUN: | FileCheck %s

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
// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK-SAME:                          [1, %[[MIN_DIM]]] [1, 1]
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = gml_st.materialize %[[INPUT]]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map
// CHECK-NEXT:     ins(%[[INPUT_SLICE]] : tensor<1x?xf32>)
// CHECK-NEXT:     outs(%[[INIT_SLICE]] : tensor<1x?xf32>)
// CHECK-NEXT:     (%[[IN_ELEM:.*]]: f32) {
// CHECK-NEXT:       %[[RES_ELEM:.*]] = math.absf %[[IN_ELEM]] : f32
// CHECK-NEXT:       linalg.yield %[[RES_ELEM]] : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   gml_st.set_yield %[[MAPPED]] into %[[INIT]][%[[TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]

// -----

func.func @map_broadcast_fuse(%arg0: tensor<?xf32>, %arg1: tensor<?x?x?xf32>,
                              %init0: tensor<?xf32>,
                              %init1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %abs = linalg.map
         ins(%arg0:tensor<?xf32>)
         outs(%init0:tensor<?xf32>)
         (%input_elem: f32) {
           %0 = math.absf %input_elem: f32
           linalg.yield %0: f32
         }

  %bcast = linalg.broadcast
           ins(%abs : tensor<?xf32>)
           outs(%init1 : tensor<?x?x?xf32>)
           dimensions = [1, 2]

  %mapped = linalg.map
            ins(%bcast, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
            outs(%init1:tensor<?x?x?xf32>)
            (%lhs: f32, %rhs: f32) {
              %0 = arith.addf %lhs, %rhs: f32
              linalg.yield %0: f32
            }

  func.return %mapped : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @map_broadcast_fuse(
// CHECK-SAME:      %[[ARG0:[0-9a-zA-Z]*]]: tensor<?xf32>,
// CHECK-SAME:      %[[ARG1:[0-9a-zA-Z]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:      %[[INIT0:[0-9a-zA-Z]*]]: tensor<?xf32>,
// CHECK-SAME:      %[[INIT1:[0-9a-zA-Z]*]]: tensor<?x?x?xf32>)

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8

// CHECK-DAG:  %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:  %[[DIM_1:.*]] = tensor.dim %[[INIT1]], %[[C1]]
// CHECK-DAG:  %[[DIM_2:.*]] = tensor.dim %[[INIT1]], %[[C2]]

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel
// CHECK-SAME:     (%[[I:.*]], %[[J:.*]], %[[K:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]], %[[C0]]) to
// CHECK-SAME:     (%[[DIM_0]], %[[DIM_1]], %[[DIM_2]])
// CHECK-SAME:     step (%[[C1]], %[[C1]], %[[C8]]) {
// CHECK:        %[[MIN_DIM:.*]] = affine.min #map(%[[K]])[%[[DIM_2]]]
// CHECK-DAG:    %[[ARG0_TILE:.*]] = gml_st.tile [%[[I]]]
// CHECK-DAG:    %[[INIT1_TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[K]]]
// CHECK-DAG:    %[[ARG0_SLICE:.*]] = gml_st.materialize %[[ARG0]]
// CHECK-DAG:    %[[INIT0_SLICE:.*]] = gml_st.materialize %[[INIT0]]

// CHECK:        %[[ABS:.*]] = linalg.map
// CHECK-NEXT:     ins(%[[ARG0_SLICE]]
// CHECK-NEXT:     outs(%[[INIT0_SLICE]]

// CHECK:        %[[INIT1_SLICE:.*]] = gml_st.materialize %[[INIT1]]
// CHECK:        %[[BCAST:.*]] = linalg.broadcast
// CHECK-NEXT:     ins(%[[ABS]]
// CHECK-NEXT:     outs(%[[INIT1_SLICE]]
// CHECK:        %[[ARG1_SLICE:.*]] = gml_st.materialize %[[ARG1]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map
// CHECK-NEXT:     ins(%[[BCAST]], %[[ARG1_SLICE]] : tensor<1x1x?xf32>
// CHECK-NEXT:     outs(%[[INIT1_SLICE]] : tensor<1x1x?xf32>)
// CHECK:        gml_st.set_yield %[[MAPPED]] into %[[INIT1]][%[[INIT1_TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]
