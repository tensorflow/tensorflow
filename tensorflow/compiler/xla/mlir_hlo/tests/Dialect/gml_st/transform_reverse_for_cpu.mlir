// RUN: mlir-hlo-opt %s -xla-cpu-transform-reverse="vector-size=8" \
// RUN: --split-input-file | FileCheck %s

func.func @reverse_static_perfect_tiles(
  %input: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %res = thlo.reverse
         ins(%input: tensor<64xf32>)
         outs(%init: tensor<64xf32>)
         reverse_dimensions = [0]
  func.return %res : tensor<64xf32>
}

// CHECK-LABEL: @reverse_static_perfect_tiles(
//  CHECK-SAME: %[[IN:.*]]: tensor<64xf32>, %[[INIT:.*]]: tensor<64xf32>
//       CHECK:   %[[PARALLEL:.*]] = gml_st.parallel (%[[IDX:.*]]) = 
//       CHECK:     %[[TEMP:.*]] = arith.subi
//       CHECK:     %[[IN_IDX:.*]] = arith.subi %[[TEMP]]
//   CHECK-DAG:     %[[IN_SLICE:.*]] = gml_st.materialize %[[IN]] [%[[IN_IDX]]]
//   CHECK-DAG:     %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]] [%[[IDX]]]
//  CHECK-NEXT:     %[[REVERSED:.*]] = thlo.reverse
//  CHECK-SAME:       ins(%[[IN_SLICE]] : tensor<8xf32>)
//  CHECK-SAME:       outs(%[[INIT_SLICE]] : tensor<8xf32>)
//  CHECK-NEXT:     %[[TILE:.*]] = gml_st.tile [%[[IDX]]] [8] [1]
//  CHECK-NEXT:   gml_st.set_yield %[[REVERSED]] into %[[INIT]][%[[TILE]]]
//       CHECK:   return %[[PARALLEL]]

// -----

func.func @reverse_dynamic(
  %input: tensor<?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %res = thlo.reverse
         ins(%input: tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         reverse_dimensions = [0, 1]
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: @reverse_dynamic(
//  CHECK-SAME: %[[IN:.*]]: tensor<?x?xf32>, %[[INIT:.*]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8
//       CHECK:   %[[DIM0:.*]] = tensor.dim %[[INIT]], %[[C0]]
//       CHECK:   %[[DIM1:.*]] = tensor.dim %[[INIT]], %[[C1]]
//       CHECK:   %[[END_IDX1:.*]] = affine.apply #map()[%[[DIM1]]]

//       CHECK:   %[[PERF_PARALLEL:.*]] = gml_st.parallel (%[[PERF_IDX0:.*]], %[[PERF_IDX1:.*]]) =
//  CHECK-SAME:   (%[[C0]], %[[C0]]) to (%[[DIM0]], %[[END_IDX1]]) step (%[[C1]], %[[C8]])
//       CHECK:     %[[PERF_REVERSED:.*]] = thlo.reverse
//  CHECK-SAME:       ins(%{{.*}} : tensor<1x?xf32>) outs(%{{.*}} : tensor<1x?xf32>)

//       CHECK:   %[[REM_PARALLEL:.*]] = gml_st.parallel (%[[REM_IDX0:.*]], %[[REM_IDX1:.*]]) =
//  CHECK-SAME:   (%[[C0]], %[[END_IDX1]]) to (%[[DIM0]], %[[DIM1]]) step (%[[C1]], %[[C8]])
//       CHECK:     %[[REM_END_IDX1:.*]] = affine.apply #map1(%[[REM_IDX1]])[%[[DIM1]]]
//       CHECK:     %[[DIM2:.*]] = tensor.dim %[[REM_INIT_SLICE:.*]], %[[C1]] : tensor<1x?xf32>

//       CHECK:     %[[INNER_PARALLEL:.*]] = gml_st.parallel (%[[INNER_IDX0:.*]], %[[INNER_IDX1:.*]]) =
//  CHECK-SAME:     (%[[C0]], %[[C0]]) to (%[[C1]], %[[DIM2]]) step (%[[C1]], %[[C1]])
//       CHECK:       %[[INNER_REVERSED:.*]] = thlo.reverse
//  CHECK-SAME:         ins(%{{.*}} : tensor<1x1xf32>) outs(%{{.*}} : tensor<1x1xf32>)

//       CHECK:   return %[[REM_PARALLEL]]
