// RUN: mlir-hlo-opt %s --gml-deprecated-tiling="tile-sizes=[16,8],[4,1]" --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-PERFECT

// RUN: mlir-hlo-opt %s --gml-deprecated-tiling="tile-sizes=[17,9],[3,3]" --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-IMPERFECT

func.func @identity(%arg: tensor<64x32xf32>) -> tensor<64x32xf32> {
  return %arg :  tensor<64x32xf32>
}

// CHECK-PERFECT-LABEL: @identity
// CHECK-PERFECT-SAME:  %[[ARG:.*]]: tensor<64x32xf32>
// CHECK-PERFECT:       %[[INIT:.*]] = linalg.init_tensor [64, 32]
// CHECK-PERFECT:       %[[SPACE:.*]] = gml_st.space [64, 32]
// CHECK-PERFECT:       %[[C0:.*]] = arith.constant 0
// CHECK-PERFECT:       %[[C64:.*]] = arith.constant 64
// CHECK-PERFECT:       %[[C32:.*]] = arith.constant 32
// CHECK-PERFECT:       %[[C16:.*]] = arith.constant 16
// CHECK-PERFECT:       %[[C8:.*]] = arith.constant 8
// CHECK-PERFECT:       %[[RES:.*]] = gml_st.parallel
// CHECK-PERFECT-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-PERFECT-SAME:      to (%[[C64]], %[[C32]])
// CHECK-PERFECT-SAME:      step (%[[C16]], %[[C8]])
// CHECK-PERFECT:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [16, 8] [1, 1]
// CHECK-PERFECT:         %[[MED_ARG:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
// CHECK-PERFECT:         %[[MED_INIT:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// CHECK-PERFECT:         %[[INNER_SPACE:.*]] = gml_st.space [16, 8]
// CHECK-PERFECT:         %[[C4:.*]] = arith.constant 4
// CHECK-PERFECT:         %[[C1:.*]] = arith.constant 1
// CHECK-PERFECT:         %[[INNER_RES:.*]] = gml_st.parallel
// CHECK-PERFECT-SAME:        (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[C0]], %[[C0]])
// CHECK-PERFECT-SAME:        to (%[[C16]], %[[C8]])
// CHECK-PERFECT-SAME:        step (%[[C4]], %[[C1]])
// CHECK-PERFECT:           %[[INNER_TILE:.*]] = gml_st.tile %[[INNER_SPACE]] [%[[ARG3]], %[[ARG4]]] [4, 1] [1, 1]
// CHECK-PERFECT:           %[[INNER_MED_ARG:.*]] = gml_st.materialize %[[MED_ARG]][%[[INNER_TILE]]]
// CHECK-PERFECT:           gml_st.set_yield %[[INNER_MED_ARG]] into %[[MED_INIT]][%[[INNER_TILE]]]
// CHECK-PERFECT:         gml_st.set_yield %[[INNER_RES]] into %[[INIT]][%[[TILE]]]
// CHECK-PERFECT:       return %[[RES]]

// CHECK-IMPERFECT-LABEL: func.func @identity
// CHECK-IMPERFECT-SAME:  %[[ARG:.*]]: tensor<64x32xf32>
// CHECK-IMPERFECT:       %[[INIT:.*]] = linalg.init_tensor [64, 32]
// CHECK-IMPERFECT:       %[[SPACE:.*]] = gml_st.space [64, 32]
// CHECK-IMPERFECT:       %[[C0:.*]] = arith.constant 0
// CHECK-IMPERFECT:       %[[C64:.*]] = arith.constant 64
// CHECK-IMPERFECT:       %[[C32:.*]] = arith.constant 32
// CHECK-IMPERFECT:       %[[C17:.*]] = arith.constant 17
// CHECK-IMPERFECT:       %[[C9:.*]] = arith.constant 9
// CHECK-IMPERFECT:       %[[RES:.*]] = gml_st.parallel
// CHECK-IMPERFECT-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-IMPERFECT-SAME:      to (%[[C64]], %[[C32]])
// CHECK-IMPERFECT-SAME:      step (%[[C17]], %[[C9]])
// CHECK-IMPERFECT:         %[[SUBI:.*]] = arith.subi %[[C64]], %[[I]]
// CHECK-IMPERFECT:         %[[SELECT:.*]] = arith.minsi %[[C17]], %[[SUBI]]
// CHECK-IMPERFECT:         %[[SUBI_0:.*]] = arith.subi %[[C32]], %[[J]]
// CHECK-IMPERFECT:         %[[SELECT_0:.*]] = arith.minsi %[[C9]], %[[SUBI_0]]
// CHECK-IMPERFECT:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[SELECT]], %[[SELECT_0]]] [1, 1]
// CHECK-IMPERFECT:         %[[MED_ARG:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
// CHECK-IMPERFECT:         %[[MED_INIT:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// CHECK-IMPERFECT:         %[[DIM:.*]] = tensor.dim %[[MED_ARG]], %[[C0]]
// CHECK-IMPERFECT:         %[[C1:.*]] = arith.constant 1
// CHECK-IMPERFECT:         %[[DIM_0:.*]] = tensor.dim %[[MED_ARG]], %[[C1]]
// CHECK-IMPERFECT:         %[[INNER_SPACE:.*]] = gml_st.space [%[[DIM]], %[[DIM_0]]]
// CHECK-IMPERFECT:         %[[C3:.*]] = arith.constant 3
// CHECK-IMPERFECT:         %[[INNER_RES:.*]] = gml_st.parallel
// CHECK-IMPERFECT-SAME:        (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[C0]], %[[C0]])
// CHECK-IMPERFECT-SAME:        to (%[[DIM]], %[[DIM_0]])
// CHECK-IMPERFECT-SAME:        step (%[[C3]], %[[C3]])
// CHECK-IMPERFECT:           %[[SUBI_1:.*]] = arith.subi %[[DIM]], %[[ARG3]]
// CHECK-IMPERFECT:           %[[SELECT_1:.*]] = arith.minsi %[[C3]], %[[SUBI_1]]
// CHECK-IMPERFECT:           %[[SUBI_2:.*]] = arith.subi %[[DIM_0]], %[[ARG4]]
// CHECK-IMPERFECT:           %[[SELECT_2:.*]] = arith.minsi %[[C3]], %[[SUBI_2]]
// CHECK-IMPERFECT:           %[[INNER_TILE:.*]] = gml_st.tile %[[INNER_SPACE]] [%[[ARG3]], %[[ARG4]]] [%[[SELECT_1]], %[[SELECT_2]]] [1, 1]
// CHECK-IMPERFECT:           %[[INNER_MED_ARG:.*]] = gml_st.materialize %[[MED_ARG]][%[[INNER_TILE]]]
// CHECK-IMPERFECT:           gml_st.set_yield %[[INNER_MED_ARG]] into %[[MED_INIT]][%[[INNER_TILE]]]
// CHECK-IMPERFECT:         gml_st.set_yield %[[INNER_RES]] into %[[INIT]][%[[TILE]]]
// CHECK-IMPERFECT:       return %[[RES]]
