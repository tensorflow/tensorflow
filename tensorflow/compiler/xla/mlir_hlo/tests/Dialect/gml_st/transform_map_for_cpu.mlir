// RUN: mlir-hlo-opt %s --gml-st-cpu-transform-map="tile-size=8" \
// RUN: --split-input-file | FileCheck %s

func.func @map_unary(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                  -> tensor<?x?xf32> {
  %abs = linalg.map { math.absf }
           ins(%input:tensor<?x?xf32>)
           outs(%init:tensor<?x?xf32>)

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
// CHECK-SAME:     step (%[[C1]], %[[C8]]) outs (%[[INIT_:.*]] = %[[INIT]]
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[INPUT]]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map { math.absf }
// CHECK-SAME:     ins(%[[INPUT_SLICE]] : tensor<1x?xf32>)
// CHECK-SAME:     outs(%[[INIT_SLICE]] : tensor<1x?xf32>)
// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[MAIN_I]], %[[MAIN_J]]]
// CHECK-SAME:                          [1, %[[C8]]] [1, 1]
// CHECK-NEXT:   gml_st.set_yield %[[MAPPED]] into %[[INIT_]][%[[TILE]]]
// CHECK-NEXT: }

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[MAP_DIM_1]]) to (%[[DIM_0]], %[[DIM_1]])
// CHECK-SAME:     step (%[[C1]], %[[C8]])
// CHECK-SAME:     outs (%[[MAIN_PAR_:.*]] = %[[MAIN_PAR]]
// CHECK:        %[[MAP_DIM:.*]] = affine.apply #{{.*}}(%[[J]])[%[[DIM_1]]]
// CHECK-NEXT:   %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[INPUT]]
// CHECK-NEXT:   %[[INIT_SLICE:.*]] = tensor.extract_slice %[[MAIN_PAR_]]

// CHECK:        %[[RESULT1:.*]] = gml_st.parallel (%[[I1:.*]], %[[J1:.*]]) =
// CHECK-SAME:       outs (%[[INIT_SLICE_:.*]] = %[[INIT_SLICE]]
// CHECK-NEXT:     %[[INPUT_SLICE1:.*]] = tensor.extract_slice %[[INPUT_SLICE]]
// CHECK-NEXT:     %[[INIT_SLICE1:.*]] = tensor.extract_slice %[[INIT_SLICE_]]
// CHECK-NEXT:     %[[MAPPED:.*]] = linalg.map { math.absf }
// CHECK-SAME:       ins(%[[INPUT_SLICE1]] : tensor<1x1xf32>)
// CHECK-SAME:       outs(%[[INIT_SLICE1]] : tensor<1x1xf32>)
// CHECK-NEXT:     %[[TILE1:.*]] = gml_st.tile [%[[I1]], %[[J1]]]
// CHECK-SAME:                                 [1, 1] [1, 1]
// CHECK-NEXT:     gml_st.set_yield %[[MAPPED]]
// CHECK-SAME:         into %[[INIT_SLICE_]][%[[TILE1]]]
// CHECK-NEXT:   }

// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK-SAME:                          [1, %[[MAP_DIM]]] [1, 1]
// CHECK-NEXT:   gml_st.set_yield %[[RESULT1]] into %[[MAIN_PAR_]][%[[TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]

// -----

func.func @map_broadcast_fuse(%arg0: tensor<?xf32>, %arg1: tensor<?x?x?xf32>,
                              %init0: tensor<?xf32>,
                              %init1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %abs = linalg.map { math.absf }
           ins(%arg0:tensor<?xf32>)
           outs(%init0:tensor<?xf32>)

  %bcast = linalg.broadcast
             ins(%abs : tensor<?xf32>)
             outs(%init1 : tensor<?x?x?xf32>)
             dimensions = [1, 2]

  %mapped = linalg.map { arith.addf }
              ins(%bcast, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
              outs(%init1:tensor<?x?x?xf32>)

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
// CHECK-SAME:     step (%[[C1]], %[[C1]], %[[C8]])
// CHECK-SAME:     outs (%[[INIT1_:.*]] = %[[INIT1]]:
// CHECK-DAG:    %[[ARG0_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[INIT0_SLICE:.*]] = tensor.extract_slice %[[INIT0]]

// CHECK:        %[[ABS:.*]] = linalg.map
// CHECK-SAME:     ins(%[[ARG0_SLICE]]
// CHECK-SAME:     outs(%[[INIT0_SLICE]]

// CHECK:        %[[INIT1_SLICE:.*]] = tensor.extract_slice %[[INIT1]]
// CHECK:        %[[BCAST:.*]] = linalg.broadcast
// CHECK-SAME:     ins(%[[ABS]]
// CHECK-SAME:     outs(%[[INIT1_SLICE]]
// CHECK:        %[[ARG1_SLICE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK:        %[[INIT1_SLICE_:.*]] = tensor.extract_slice %[[INIT1_]]
// CHECK-NEXT:   %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:     ins(%[[BCAST]], %[[ARG1_SLICE]] : tensor<1x1x?xf32>
// CHECK-SAME:     outs(%[[INIT1_SLICE_]] : tensor<1x1x?xf32>)
// CHECK:        %[[INIT1_TILE:.*]] = gml_st.tile [%[[MAIN_I]], %[[MAIN_J]], %[[MAIN_K]]]
// CHECK:        gml_st.set_yield %[[MAPPED]] into %[[INIT1_]][%[[INIT1_TILE]]]
// CHECK-NEXT: }

// CHECK-NEXT: %[[RESULT:.*]] = gml_st.parallel
// CHECK-SAME:     (%[[I:.*]], %[[J:.*]], %[[K:.*]]) =
// CHECK-SAME:     (%[[C0]], %[[C0]], %[[MAP_DIM_2]]) to
// CHECK-SAME:     (%[[DIM_0]], %[[DIM_1]], %[[DIM_2]])
// CHECK-SAME:     step (%[[C1]], %[[C1]], %[[C8]])
// CHECK-SAME:     outs (%[[MAIN_PAR_:.*]] = %[[MAIN_PAR]]:
// CHECK:        %[[MAP_DIM:.*]] = affine.apply #{{.*}}(%[[K]])[%[[DIM_2]]]
// CHECK-DAG:    %[[ARG1_SLICE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK-DAG:    %[[INIT0_SLICE:.*]] = tensor.extract_slice %[[INIT0]]
// CHECK-DAG:    %[[MAIN_PAR_SLICE:.*]] = tensor.extract_slice %[[MAIN_PAR]]
// CHECK-DAG:    %[[MAIN_PAR_SLICE_:.*]] = tensor.extract_slice %[[MAIN_PAR_]]


// CHECK:        %[[RESULT1:.*]] = gml_st.parallel
// CHECK-SAME:       (%[[I1:.*]], %[[J1:.*]], %[[K1:.*]]) =
// CHECK-SAME:       outs (%[[OUT_:.*]] = %[[MAIN_PAR_SLICE_]]:
// CHECK-DAG:      %[[ARG0_SLICE1:.*]] = tensor.extract_slice %[[ARG0_SLICE]][%[[I1]]]
// CHECK-DAG:      %[[INIT0_SLICE1:.*]] = tensor.extract_slice %[[INIT0_SLICE]][%[[I1]]]
// CHECK:          %[[ABS1:.*]] = linalg.map
// CHECK-SAME:       ins(%[[ARG0_SLICE1]]
// CHECK-SAME:       outs(%[[INIT0_SLICE1]]

// CHECK:          %[[INIT1_SLICE1:.*]] = tensor.extract_slice %[[INIT1_SLICE]]
// CHECK:          %[[BCAST:.*]] = linalg.broadcast
// CHECK-SAME:       ins(%[[ABS1]]
// CHECK-SAME:       outs(%[[INIT1_SLICE1]]
// CHECK:          %[[ARG1_SLICE1:.*]] = tensor.extract_slice %[[ARG1_SLICE]]
// CHECK:          %[[OUT_SLICE_:.*]] = tensor.extract_slice %[[OUT_]]
// CHECK-NEXT:     %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:       ins(%[[BCAST]], %[[ARG1_SLICE1]] : tensor<1x1x1xf32>
// CHECK-SAME:       outs(%[[OUT_SLICE_]] : tensor<1x1x1xf32>)
// CHECK-DAG:      %[[INIT1_TILE1:.*]] = gml_st.tile [%[[I1]], %[[J1]], %[[K1]]]
// CHECK:          gml_st.set_yield %[[MAPPED]] into %[[OUT_]][%[[INIT1_TILE1]]]
// CHECK-DAG:    %[[INIT1_TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[K]]]
// CHECK:        gml_st.set_yield %[[RESULT1]] into %[[MAIN_PAR_]][%[[INIT1_TILE]]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]

// -----

func.func @map_non_unique_users(%arg: tensor<?x?xf32>,
                              %init: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %exp = linalg.map { math.exp }
           ins(%arg: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)

  %mul = linalg.map { arith.mulf }
           ins(%exp, %exp: tensor<?x?xf32>, tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)

  %abs = linalg.map { math.absf }
           ins(%mul: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)

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
