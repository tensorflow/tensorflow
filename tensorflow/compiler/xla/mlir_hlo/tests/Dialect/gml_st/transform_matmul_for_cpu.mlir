// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=CHECK,TRANSFORMED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=MARKED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="lower-to-mmt4d=true" | FileCheck %s --check-prefixes=MMT4D

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

// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK:        %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:       -> tensor<8x4xf32>
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.set_yield %[[FOR]]

// -----

// MMT4D-LABEL:    func @matmul_static(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c8, %c8) step (%c8, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1)
// MMT4D:                    linalg.mmt4d

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

// TRANSFORMED-LABEL: func @matmul(
// TRANSFORMED-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// TRANSFORMED-DAG:     %[[C0:.*]] = arith.constant 0 : index
// TRANSFORMED:         %[[INIT:.*]] = tensor.empty

// TRANSFORMED:         %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// TRANSFORMED:           %[[MAIN_SLICE:.*]] = gml_st.materialize %[[INIT]]
// TRANSFORMED:           %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
// TRANSFORMED:           %[[MAIN_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
// TRANSFORMED:             %[[MAIN_PAR_MAIN_FOR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED-NEXT:        gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_MATMUL]]
// TRANSFORMED:           %[[REM_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[KUB]]) {{.*}} outs ({{.*}} = %[[MAIN_FOR]]:
// TRANSFORMED:             %[[MAIN_PAR_REM_FOR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED-NEXT:        gml_st.set_yield %[[MAIN_PAR_REM_FOR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_FOR]]

// TRANSFORMED:         %[[REM_RHS_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// TRANSFORMED:           %[[REM_RHS_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
// TRANSFORMED:           %[[REM_RHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_RHS_SLICE]]
// TRANSFORMED:           %[[REM_RHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_RHS_FILL]]:
// TRANSFORMED:             %[[REM_RHS_PAR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED-NEXT:        gml_st.set_yield %[[REM_RHS_PAR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_RHS_FOR]]

// TRANSFORMED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// TRANSFORMED:           %[[REM_LHS_SLICE:.*]] = gml_st.materialize %[[REM_RHS_PAR]]
// TRANSFORMED:           %[[REM_LHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_LHS_SLICE]]
// TRANSFORMED:           %[[REM_LHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_LHS_FILL]]:
// TRANSFORMED:             %[[REM_LHS_PAR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED-NEXT:        gml_st.set_yield %[[REM_LHS_PAR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_LHS_FOR]]

// -----

// MARKED-LABEL: func @matmul(

// MARKED:         %[[C0:.*]] = arith.constant 0 : index
// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) step
// MARKED:           __perfectly_tiled_loop_label__
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[KUB]])

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])

// -----

// MMT4D-LABEL:    func @matmul(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c8, %c8) step (%c8, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1)
// MMT4D:                    linalg.mmt4d
