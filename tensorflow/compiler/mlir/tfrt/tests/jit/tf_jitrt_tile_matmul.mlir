// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -xla-cpu-transform-matmul="tile-sizes=8,4,2" \
// RUN: | FileCheck %s

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
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

// CHECK-LABEL: func @matmul(
// CHECK-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[INIT:.*]] = tensor.empty

// CHECK:         %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// CHECK:           %[[MAIN_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK:           %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
// CHECK:           %[[MAIN_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
// CHECK:             %[[MAIN_PAR_MAIN_FOR_MATMUL:.*]] = linalg.matmul
// CHECK-NEXT:        gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_MATMUL]]

// CHECK:           %[[REM_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[KUB]]) {{.*}} outs ({{.*}} = %[[MAIN_FOR]]:
// CHECK:             %[[MAIN_PAR_REM_FOR_MATMUL:.*]] = linalg.matmul
// CHECK-NEXT:        gml_st.set_yield %[[MAIN_PAR_REM_FOR_MATMUL]]
// CHECK:           gml_st.set_yield %[[REM_FOR]]

// CHECK:         %[[REM_LHS_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// CHECK:           %[[REM_LHS_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
// CHECK:           %[[REM_LHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_LHS_SLICE]]
// CHECK:           %[[REM_LHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_LHS_FILL]]:
// CHECK:             %[[REM_LHS_PAR_MATMUL:.*]] = linalg.matmul
// CHECK-NEXT:        gml_st.set_yield %[[REM_LHS_PAR_MATMUL]]
// CHECK:           gml_st.set_yield %[[REM_LHS_FOR]]

// CHECK:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// CHECK:           %[[REM_RHS_SLICE:.*]] = gml_st.materialize %[[REM_LHS_PAR]]
// CHECK:           %[[REM_RHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_RHS_SLICE]]
// CHECK:           %[[REM_RHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_RHS_FILL]]:
// CHECK:             %[[REM_RHS_PAR_MATMUL:.*]] = linalg.matmul
// CHECK-NEXT:        gml_st.set_yield %[[REM_RHS_PAR_MATMUL]]
// CHECK:           gml_st.set_yield %[[REM_RHS_FOR]]
