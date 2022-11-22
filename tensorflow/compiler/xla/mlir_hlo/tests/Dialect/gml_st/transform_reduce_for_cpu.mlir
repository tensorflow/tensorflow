// RUN: mlir-hlo-opt %s -xla-cpu-transform-reduce="tile-sizes=4,2" --split-input-file | FileCheck %s --check-prefixes=CHECK,PEELED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-reduce="tile-sizes=4,2" --split-input-file | FileCheck %s --check-prefixes=MARKED

func.func @reduce_add_static(%input: tensor<100x10xf32>,
                        %output: tensor<10xf32>) -> tensor<10xf32> {
  %res = linalg.reduce ins(%input: tensor<100x10xf32>)
                     outs(%output: tensor<10xf32>)
                     dimensions = [0]
          (%in: f32, %init: f32) {
            %0 = arith.addf %in, %init : f32
            linalg.yield %0 : f32
          }
  return %res : tensor<10xf32>
}

// CHECK-LABEL:     func @reduce_add_static(
//  CHECK-SAME:       %[[IN:.*]]: tensor<100x10xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<10xf32>)
//  CHECK-SAME:       -> tensor<10xf32> {

//       CHECK:       %[[C0:.*]] = arith.constant 0 : index

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])

//       CHECK:         %[[OUT_DIM:.*]] = tensor.dim %[[TILED_IN:.*]], %[[C1:.*]]
//       CHECK:         %[[FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]])

//       CHECK:           %[[REDUCED:.*]] = linalg.reduce
//  CHECK-NEXT:             ins(%[[IN:.*]] : tensor<2x?xf32>)
//  CHECK-NEXT:             outs(%[[OUT:.*]] : tensor<?xf32>)
//  CHECK-NEXT:             dimensions = [0]

//       CHECK:           gml_st.set_yield %[[REDUCED]]
//       CHECK:         gml_st.set_yield %[[FOR]]

// -----

func.func @reduce_mulf(%input: tensor<?x?xf32>,
                      %output: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %output, %c0 : tensor<?xf32>
  %1 = tensor.empty(%0) : tensor<?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  %res = linalg.reduce ins(%input: tensor<?x?xf32>)
                     outs(%2: tensor<?xf32>)
                     dimensions = [1]
          (%in: f32, %init: f32) {
            %mulf = arith.mulf %in, %init : f32
            linalg.yield %mulf : f32
          }
  return %res : tensor<?xf32>
}

// PEELED-LABEL:     func @reduce_mulf(
//  PEELED-SAME:       %[[IN:.*]]: tensor<?x?xf32>,
//  PEELED-SAME:       %[[OUT:.*]]: tensor<?xf32>)
//  PEELED-SAME:       -> tensor<?xf32> {

//   PEELED-DAG:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   PEELED-DAG:       %[[C0:.*]] = arith.constant 0 : index
//   PEELED-DAG:       %[[C1:.*]] = arith.constant 1 : index
//       PEELED:       %[[INIT:.*]] = tensor.empty
//       PEELED:       %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]]

//       PEELED:       %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[IUB:.*]]) step
//       PEELED:         %[[MAIN_SLICE:.*]] = gml_st.materialize %[[INIT]]
//       PEELED:         %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
//       PEELED:         %[[MAIN_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]]) to (%[[JUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
//       PEELED:           %[[MAIN_PAR_MAIN_FOR_REDUCE:.*]] = linalg.reduce
//       PEELED:           gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_REDUCE]]
//       PEELED:         %[[REM_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[JUB]]) {{.*}} outs ({{.*}} = %[[MAIN_FOR]]:
//       PEELED:           %[[MAIN_PAR_REM_FOR_REDUCE:.*]] = linalg.reduce
//       PEELED:           gml_st.set_yield %[[MAIN_PAR_REM_FOR_REDUCE]]
//       PEELED:         gml_st.set_yield %[[REM_FOR]]

//       PEELED:       %[[REM_PAR:.*]] = gml_st.parallel (%[[I:.*]]) = (%[[IUB]])
//       PEELED:         %[[REM_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
//       PEELED:         %[[REM_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_SLICE]]
//       PEELED:         %[[REM_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_FILL]]:
//       PEELED:           %[[REM_PAR_REDUCE:.*]] = linalg.reduce
//       PEELED:           gml_st.set_yield %[[REM_PAR_REDUCE]]
//       PEELED:         gml_st.set_yield %[[REM_FOR]]

// -----

// MARKED-LABEL:     func @reduce_mulf(
//       MARKED:       %[[C0:.*]] = arith.constant 0 : index
//       MARKED:       gml_st.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[IUB:.*]]) step
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[C0]]) to (%[[JUB:.*]]) step
//       MARKED:         } {__internal_peeled_marker__ = true}
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[JUB]])
//       MARKED:         } {__internal_peeled_marker__ = true}
//       MARKED:       } {__internal_peeled_marker__ = true}

//       MARKED:       gml_st.parallel (%[[I:.*]]) = (%[[IUB]])
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[C0]])
//       MARKED:         } {__internal_peeled_marker__ = true}
//       MARKED:       } {__internal_peeled_marker__ = true}
