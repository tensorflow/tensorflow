// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -xla-cpu-transform-matmul="tile-sizes=0,0,0" \
// RUN: | FileCheck %s --check-prefix=TILE-EMPTY

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

// TILE-EMPTY-LABEL: func @matmul(
// TILE-EMPTY-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// TILE-EMPTY-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// TILE-EMPTY-DAG:   %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// TILE-EMPTY-DAG:   %[[C0:.*]] = arith.constant 0 : index
// TILE-EMPTY-DAG:   %[[C1:.*]] = arith.constant 1 : index

// TILE-EMPTY:       %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// TILE-EMPTY:       %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// TILE-EMPTY:       %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// TILE-EMPTY:       %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]

// TILE-EMPTY:       %[[MATMUL:.*]] = linalg.matmul
// TILE-EMPTY-SAME:    ins(%[[LHS]], %[[RHS]] : [[TY_2D]], [[TY_2D]])
// TILE-EMPTY-SAME:    outs(%[[FILL]] : [[TY_2D]])

// -----

// CHECK-LABEL:      func @matmul(
// CHECK-SAME:         %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:         %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG:        %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:        %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:        %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:        %[[C8:.*]] = arith.constant 8 : index

// CHECK:            %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// CHECK:            %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// CHECK:            %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// CHECK:            %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// CHECK:            %[[LHS_ROW:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// CHECK:            %[[LHS_COL:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]
// CHECK:            %[[RHS_COL:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// CHECK:            gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:         to (%[[LHS_ROW]], %[[RHS_COL]]) step (%[[C8]], %[[C4]])

// CHECK:            %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0]
// CHECK:            %[[LHS_SLICE:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]

// CHECK:            %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]]
// CHECK:            %[[RHS_SLICE:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]

// CHECK:            %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK:            %[[OUT_SLICE:.*]] = gml_st.materialize %[[FILL]][%[[OUT_TILE]]]

// CHECK:            %[[LHS_SUB_ROW:.*]] = tensor.dim %[[LHS_SLICE]], %[[C0]] : [[TY_2D]]
// CHECK:            %[[LHS_SUB_COL:.*]] = tensor.dim %[[LHS_SLICE]], %[[C1]] : [[TY_2D]]
// CHECK:            %[[RHS_SUB_COL:.*]] = tensor.dim %[[RHS_SLICE]], %[[C1]] : [[TY_2D]]
// CHECK:            %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK-SAME:         to (%[[LHS_SUB_COL]]) step (%[[C2]])
// CHECK-SAME:         outs (%[[OUT_SUB_ARG:.*]] = %[[OUT_SLICE]]: [[TY_2D]])

// CHECK:            %[[LHS_SUB_TILE:.*]] = gml_st.tile [0, %[[K]]]
// CHECK:            %[[LHS_SUB_SLICE:.*]] = gml_st.materialize %[[LHS_SLICE]][%[[LHS_SUB_TILE]]]

// CHECK:            %[[RHS_SUB_TILE:.*]] = gml_st.tile [%[[K]], 0]
// CHECK:            %[[RHS_SUB_SLICE:.*]] = gml_st.materialize %[[RHS_SLICE]][%[[RHS_SUB_TILE]]]

// CHECK:            %[[OUT_SUB_TILE:.*]] = gml_st.tile [0, 0] [%[[LHS_SUB_ROW]], %[[RHS_SUB_COL]]]
// CHECK:            %[[OUT_SUB_SLICE:.*]] = gml_st.materialize %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// CHECK:            %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_SUB_SLICE]], %[[RHS_SUB_SLICE]] : [[TY_2D]], [[TY_2D]])
// CHECK:              outs(%[[OUT_SUB_SLICE]] : [[TY_2D]])

// CHECK-NEXT:       gml_st.set_yield %[[MATMUL]] into %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// CHECK:            gml_st.set_yield %[[FOR]] into %[[FILL]][%[[OUT_TILE]]]
