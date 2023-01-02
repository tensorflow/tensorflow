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
// CHECK-DAG:  %[[MAP_DIM_1:.*]] = affine.apply {{.*}}%[[DIM_1]]

// CHECK-NEXT: %[[MAIN_PAR:.*]] = gml_st.parallel (%[[MAIN_I:.*]], %[[MAIN_J:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]]) to (%[[DIM_0]], %[[MAP_DIM_1]])
// CHECK-SAME:     step (%[[C1]], %[[C8]]) {
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = gml_st.materialize %[[INPUT]]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map { math.absf }
// CHECK-SAME:     ins(%[[INPUT_SLICE]] : tensor<1x?xf32>)
// CHECK-SAME:     outs(%[[INIT_SLICE]] : tensor<1x?xf32>)
// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[MAIN_I]], %[[MAIN_J]]]
// CHECK-SAME:                          [1, %[[C8]]] [1, 1]
// CHECK-NEXT:   gml_st.set_yield %[[MAPPED]] into %[[INIT]][%[[TILE]]]
// CHECK-NEXT: }

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[MAP_DIM_1]]) to (%[[DIM_0]], %[[DIM_1]])
// CHECK-SAME:     step (%[[C1]], %[[C8]]) {
// CHECK:        %[[MAP_DIM:.*]] = affine.apply #{{.*}}(%[[J]])[%[[DIM_1]]]
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = gml_st.materialize %[[INPUT]]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]

// CHECK:        %[[RESULT1:.*]] = gml_st.parallel (%[[I1:.*]], %[[J1:.*]]) =
// CHECK-NEXT:     %[[INPUT_SLICE1:.*]] = gml_st.materialize %[[INPUT_SLICE]]
// CHECK-NEXT:     %[[INIT_SLICE1:.*]] = gml_st.materialize %[[INIT_SLICE]]
// CHECK-NEXT:     %[[MAPPED:.*]] = linalg.map { math.absf }
// CHECK-SAME:       ins(%[[INPUT_SLICE1]] : tensor<1x1xf32>)
// CHECK-SAME:       outs(%[[INIT_SLICE1]] : tensor<1x1xf32>)
// CHECK-NEXT:     %[[TILE1:.*]] = gml_st.tile [%[[I1]], %[[J1]]]
// CHECK-SAME:                                 [1, 1] [1, 1]
// CHECK-NEXT:     gml_st.set_yield %[[MAPPED]]
// CHECK-SAME:         into %[[INIT_SLICE]][%[[TILE1]]]
// CHECK-NEXT:   }

// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK-SAME:                          [1, %[[MAP_DIM]]] [1, 1]
// CHECK-NEXT:   gml_st.set_yield %[[RESULT1]] into %[[MAIN_PAR]][%[[TILE]]]
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
// CHECK-DAG:  %[[MAP_DIM_2:.*]] = affine.apply {{.*}}%[[DIM_2]]

// CHECK-NEXT: %[[MAIN_PAR:.*]] = gml_st.parallel
// CHECK-SAME:     (%[[MAIN_I:.*]], %[[MAIN_J:.*]], %[[MAIN_K:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]], %[[C0]]) to
// CHECK-SAME:     (%[[DIM_0]], %[[DIM_1]], %[[MAP_DIM_2]])
// CHECK-SAME:     step (%[[C1]], %[[C1]], %[[C8]]) {
// CHECK-DAG:    %[[ARG0_SLICE:.*]] = gml_st.materialize %[[ARG0]]
// CHECK-DAG:    %[[INIT0_SLICE:.*]] = gml_st.materialize %[[INIT0]]

// CHECK:        %[[ABS:.*]] = linalg.map
// CHECK-SAME:     ins(%[[ARG0_SLICE]]
// CHECK-SAME:     outs(%[[INIT0_SLICE]]

// CHECK:        %[[INIT1_SLICE:.*]] = gml_st.materialize %[[INIT1]]
// CHECK:        %[[BCAST:.*]] = linalg.broadcast
// CHECK-SAME:     ins(%[[ABS]]
// CHECK-SAME:     outs(%[[INIT1_SLICE]]
// CHECK:        %[[ARG1_SLICE:.*]] = gml_st.materialize %[[ARG1]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:     ins(%[[BCAST]], %[[ARG1_SLICE]] : tensor<1x1x?xf32>
// CHECK-SAME:     outs(%[[INIT1_SLICE]] : tensor<1x1x?xf32>)
// CHECK:        %[[INIT1_TILE:.*]] = gml_st.tile [%[[MAIN_I]], %[[MAIN_J]], %[[MAIN_K]]]
// CHECK:        gml_st.set_yield %[[MAPPED]] into %[[INIT1]][%[[INIT1_TILE]]]
// CHECK-NEXT: }

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel
// CHECK-SAME:     (%[[I:.*]], %[[J:.*]], %[[K:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]], %[[MAP_DIM_2]]) to
// CHECK-SAME:     (%[[DIM_0]], %[[DIM_1]], %[[DIM_2]])
// CHECK-SAME:     step (%[[C1]], %[[C1]], %[[C8]]) {
// CHECK:        %[[MAP_DIM:.*]] = affine.apply #{{.*}}(%[[K]])[%[[DIM_2]]]
// CHECK-DAG:    %[[ARG0_SLICE:.*]] = gml_st.materialize %[[ARG0]] [%[[I]]]
// CHECK-DAG:    %[[INIT0_SLICE:.*]] = gml_st.materialize %[[INIT0]]
// CHECK-DAG:    %[[INIT1_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
// CHECK-SAME:     [%[[I]], %[[J]], %[[K]]]

// CHECK:        %[[ARG1_SLICE:.*]] = gml_st.materialize %[[ARG1]]

// CHECK:        %[[RESULT1:.*]] = gml_st.parallel
// CHECK-SAME:       (%[[I1:.*]], %[[J1:.*]], %[[K1:.*]]) =
// CHECK-DAG:      %[[ARG0_SLICE1:.*]] = gml_st.materialize %[[ARG0_SLICE]] [%[[I1]]]
// CHECK-DAG:      %[[INIT0_SLICE1:.*]] = gml_st.materialize %[[INIT0_SLICE]] [%[[I1]]]
// CHECK:          %[[ABS1:.*]] = linalg.map
// CHECK-SAME:       ins(%[[ARG0_SLICE1]]
// CHECK-SAME:       outs(%[[INIT0_SLICE1]]

// CHECK:          %[[INIT1_SLICE1:.*]] = gml_st.materialize %[[INIT1_SLICE]]
// CHECK:          %[[BCAST:.*]] = linalg.broadcast
// CHECK-SAME:       ins(%[[ABS1]]
// CHECK-SAME:       outs(%[[INIT1_SLICE1]]
// CHECK:          %[[ARG1_SLICE1:.*]] = gml_st.materialize %[[ARG1_SLICE]]
// CHECK-NEXT:     %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:       ins(%[[BCAST]], %[[ARG1_SLICE1]] : tensor<1x1x1xf32>
// CHECK-SAME:       outs(%[[INIT1_SLICE1]] : tensor<1x1x1xf32>)
// CHECK-DAG:      %[[INIT1_TILE1:.*]] = gml_st.tile [%[[I1]], %[[J1]], %[[K1]]]
// CHECK:          gml_st.set_yield %[[MAPPED]] into %[[INIT1_SLICE]][%[[INIT1_TILE1]]]
// CHECK-DAG:    %[[INIT1_TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[K]]]
// CHECK:        gml_st.set_yield %[[RESULT1]] into %[[MAIN_PAR]][%[[INIT1_TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]

// -----

func.func @map_non_unique_users(%arg: tensor<?x?xf32>,
                              %init: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %exp = linalg.map
         ins(%arg: tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         (%input1: f32) {
           %0 = math.exp %input1 : f32
           linalg.yield %0: f32
         }

  %mul = linalg.map
         ins(%exp, %exp: tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         (%input1: f32, %input2: f32) {
           %0 = arith.mulf %input1, %input2 : f32
           linalg.yield %0: f32
         }

  %abs = linalg.map
         ins(%mul: tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         (%input1: f32) {
           %0 = math.absf %input1 : f32
           linalg.yield %0: f32
         }
  func.return %abs : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @map_non_unique_users(
// CHECK:          gml_st.parallel
// CHECK:            math.exp
// CHECK-NOT:        math.exp
// CHECK:            arith.mulf
// CHECK:            math.absf
// CHECK:          gml_st.parallel
// CHECK:            math.exp
// CHECK-NOT:        math.exp
// CHECK:            arith.mulf
// CHECK:            math.absf
