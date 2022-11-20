// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=CHECK,PEELED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=MARKED

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_static(%arg0: tensor<128x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

// CHECK-LABEL:    func @matmul_static(
// CHECK-SAME:       %[[LHS:.*]]: tensor<128x16xf32>,
// CHECK-SAME:       %[[RHS:.*]]: tensor<16x64xf32>,
// CHECK-SAME:       %[[OUT:.*]]: tensor<128x64xf32>)

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index

// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])

// CHECK:        %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])

// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:       -> tensor<8x4xf32>

// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.set_yield %[[FOR]]

// -----

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
                  -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// PEELED-LABEL: func @matmul(
// PEELED-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// PEELED-DAG:     %[[C0:.*]] = arith.constant 0 : index
// PEELED:         %[[INIT:.*]] = tensor.empty

// PEELED:         %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// PEELED:           %[[MAIN_SLICE:.*]] = gml_st.materialize %[[INIT]]
// PEELED:           %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
// PEELED:           %[[MAIN_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
// PEELED:             %[[MAIN_PAR_MAIN_FOR_MATMUL:.*]] = linalg.matmul
// PEELED-NEXT:        gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_MATMUL]]
// PEELED:           %[[REM_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[KUB]]) {{.*}} outs ({{.*}} = %[[MAIN_FOR]]:
// PEELED:             %[[MAIN_PAR_REM_FOR_MATMUL:.*]] = linalg.matmul
// PEELED-NEXT:        gml_st.set_yield %[[MAIN_PAR_REM_FOR_MATMUL]]
// PEELED:           gml_st.set_yield %[[REM_FOR]]

// PEELED:         %[[REM_RHS_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// PEELED:           %[[REM_RHS_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
// PEELED:           %[[REM_RHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_RHS_SLICE]]
// PEELED:           %[[REM_RHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_RHS_FILL]]:
// PEELED:             %[[REM_RHS_PAR_MATMUL:.*]] = linalg.matmul
// PEELED-NEXT:        gml_st.set_yield %[[REM_RHS_PAR_MATMUL]]
// PEELED:           gml_st.set_yield %[[REM_RHS_FOR]]

// PEELED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// PEELED:           %[[REM_LHS_SLICE:.*]] = gml_st.materialize %[[REM_RHS_PAR]]
// PEELED:           %[[REM_LHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_LHS_SLICE]]
// PEELED:           %[[REM_LHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_LHS_FILL]]:
// PEELED:             %[[REM_LHS_PAR_MATMUL:.*]] = linalg.matmul
// PEELED-NEXT:        gml_st.set_yield %[[REM_LHS_PAR_MATMUL]]
// PEELED:           gml_st.set_yield %[[REM_LHS_FOR]]

// -----

// MARKED-LABEL: func @matmul(

// MARKED:         %[[C0:.*]] = arith.constant 0 : index
// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) step
// MARKED:           } {__internal_peeled_marker__ = true}
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[KUB]])
// MARKED:           } {__internal_peeled_marker__ = true}
// MARKED:         } {__internal_peeled_marker__ = true}

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])
// MARKED:           } {__internal_peeled_marker__ = true}
// MARKED:         } {__internal_peeled_marker__ = true}

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])
// MARKED:           } {__internal_peeled_marker__ = true}
// MARKED:         } {__internal_peeled_marker__ = true}
