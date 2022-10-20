// RUN: xla-cpu-opt %s -xla-cpu-transform-matmul | FileCheck %s --check-prefixes COMMON,SIZE-TWO
// RUN: xla-cpu-opt %s -xla-cpu-transform-matmul="tile-sizes=2,2,2" | FileCheck %s --check-prefixes COMMON,SIZE-TWO
// RUN: xla-cpu-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes COMMON,DIFF-SIZES

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_static(%arg0: tensor<128x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

// COMMON-LABEL:    func @matmul_static(
// COMMON-SAME:       %[[LHS:.*]]: tensor<128x16xf32>,
// COMMON-SAME:       %[[RHS:.*]]: tensor<16x64xf32>,
// COMMON:            %[[OUT:.*]]: tensor<128x64xf32>)

// COMMON-DAG:      %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:      %[[C2:.*]] = arith.constant 2 : index
// COMMON-DAG:      %[[C_LHS_COL:.*]] = arith.constant 16 : index
// COMMON-DAG:      %[[C_RHS_COL:.*]] = arith.constant 64 : index
// COMMON-DAG:      %[[C_LHS_ROW:.*]] = arith.constant 128 : index

// DIFF-SIZES-DAG:  %[[C4:.*]] = arith.constant 4 : index
// DIFF-SIZES-DAG:  %[[C8:.*]] = arith.constant 8 : index

// SIZE-TWO:        gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// SIZE-TWO-SAME:       to (%[[C_LHS_ROW]], %[[C_RHS_COL]]) step (%[[C2]], %[[C2]])

// DIFF-SIZES:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// DIFF-SIZES-SAME:     to (%[[C_LHS_ROW]], %[[C_RHS_COL]]) step (%[[C8]], %[[C4]])

// SIZE-TWO:        %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0] [2, 16]
// DIFF-SIZES:      %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0] [8, 16]
// COMMON:          %[[LHS_SLICE:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]

// SIZE-TWO:        %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]] [16, 2]
// DIFF-SIZES:      %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]] [16, 4]
// COMMON:          %[[RHS_SLICE:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]

// SIZE-TWO:        %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]] [2, 2]
// DIFF-SIZES:      %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]] [8, 4]
// COMMON:          %[[OUT_SLICE:.*]] = gml_st.materialize %[[OUT]][%[[OUT_TILE]]]

// COMMON:          %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// COMMON-SAME:       to (%[[C_LHS_COL]]) step (%[[C2]])
// COMMON-SAME:       outs (%[[OUT_SUB_ARG:.*]] = %[[OUT_SLICE]]:

// SIZE-TWO:        %[[LHS_SUB_TILE:.*]] = gml_st.tile [0, %[[K]]] [2, 2]
// DIFF-SIZES:      %[[LHS_SUB_TILE:.*]] = gml_st.tile [0, %[[K]]] [8, 2]
// COMMON:          %[[LHS_SUB_SLICE:.*]] = gml_st.materialize %[[LHS_SLICE]][%[[LHS_SUB_TILE]]]

// SIZE-TWO:        %[[RHS_SUB_TILE:.*]] = gml_st.tile [%[[K]], 0] [2, 2]
// DIFF-SIZES:      %[[RHS_SUB_TILE:.*]] = gml_st.tile [%[[K]], 0] [2, 4]
// COMMON:          %[[RHS_SUB_SLICE:.*]] = gml_st.materialize %[[RHS_SLICE]][%[[RHS_SUB_TILE]]]

// SIZE-TWO:        %[[OUT_SUB_TILE:.*]] = gml_st.tile [0, 0] [2, 2]
// DIFF-SIZES:      %[[OUT_SUB_TILE:.*]] = gml_st.tile [0, 0] [8, 4]
// COMMON:          %[[OUT_SUB_SLICE:.*]] = gml_st.materialize %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// COMMON:          %[[MATMUL:.*]] = linalg.matmul
// COMMON-SAME:       ins(%[[LHS_SUB_SLICE]], %[[RHS_SUB_SLICE]] :
// COMMON:            outs(%[[OUT_SUB_SLICE]] :

// COMMON-NEXT:     gml_st.set_yield %[[MATMUL]] into %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// COMMON:          gml_st.set_yield %[[FOR]] into %[[OUT]][%[[OUT_TILE]]]

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

// COMMON-LABEL:     func @matmul(
// COMMON-SAME:        %[[LHS:.*]]: tensor<?x?xf32>,
// COMMON-SAME:        %[[RHS:.*]]: tensor<?x?xf32>)

// COMMON-DAG:      %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// COMMON-DAG:      %[[C0:.*]] = arith.constant 0 : index
// COMMON-DAG:      %[[C1:.*]] = arith.constant 1 : index
// COMMON-DAG:      %[[C2:.*]] = arith.constant 2 : index

// DIFF-SIZES-DAG:  %[[C4:.*]] = arith.constant 4 : index
// DIFF-SIZES-DAG:  %[[C8:.*]] = arith.constant 8 : index

// COMMON:          %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// COMMON:          %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// COMMON:          %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// COMMON:          %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// COMMON:          %[[LHS_ROW:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// COMMON:          %[[LHS_COL:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]
// COMMON:          %[[RHS_COL:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// SIZE-TWO:        gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// SIZE-TWO-SAME:     to (%[[LHS_ROW]], %[[RHS_COL]]) step (%[[C2]], %[[C2]])

// DIFF-SIZES:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// DIFF-SIZES-SAME:   to (%[[LHS_ROW]], %[[RHS_COL]]) step (%[[C8]], %[[C4]])

// COMMON:          %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0]
// COMMON:          %[[LHS_SLICE:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]

// COMMON:          %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]]
// COMMON:          %[[RHS_SLICE:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]

// COMMON:          %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// COMMON:          %[[OUT_SLICE:.*]] = gml_st.materialize %[[FILL]][%[[OUT_TILE]]]

// COMMON:          %[[LHS_SUB_ROW:.*]] = tensor.dim %[[LHS_SLICE]], %[[C0]] : [[TY_2D]]
// COMMON:          %[[LHS_SUB_COL:.*]] = tensor.dim %[[LHS_SLICE]], %[[C1]] : [[TY_2D]]
// COMMON:          %[[RHS_SUB_COL:.*]] = tensor.dim %[[RHS_SLICE]], %[[C1]] : [[TY_2D]]
// COMMON:          %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// COMMON-SAME:       to (%[[LHS_SUB_COL]]) step (%[[C2]])
// COMMON-SAME:       outs (%[[OUT_SUB_ARG:.*]] = %[[OUT_SLICE]]: [[TY_2D]])

// COMMON:          %[[LHS_SUB_TILE:.*]] = gml_st.tile [0, %[[K]]]
// COMMON:          %[[LHS_SUB_SLICE:.*]] = gml_st.materialize %[[LHS_SLICE]][%[[LHS_SUB_TILE]]]

// COMMON:          %[[RHS_SUB_TILE:.*]] = gml_st.tile [%[[K]], 0]
// COMMON:          %[[RHS_SUB_SLICE:.*]] = gml_st.materialize %[[RHS_SLICE]][%[[RHS_SUB_TILE]]]

// COMMON:          %[[OUT_SUB_TILE:.*]] = gml_st.tile [0, 0] [%[[LHS_SUB_ROW]], %[[RHS_SUB_COL]]]
// COMMON:          %[[OUT_SUB_SLICE:.*]] = gml_st.materialize %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// COMMON:          %[[MATMUL:.*]] = linalg.matmul
// COMMON-SAME:       ins(%[[LHS_SUB_SLICE]], %[[RHS_SUB_SLICE]] : [[TY_2D]], [[TY_2D]])
// COMMON:            outs(%[[OUT_SUB_SLICE]] : [[TY_2D]])

// COMMON-NEXT:     gml_st.set_yield %[[MATMUL]] into %[[OUT_SUB_ARG]][%[[OUT_SUB_TILE]]]

// COMMON:          gml_st.set_yield %[[FOR]] into %[[FILL]][%[[OUT_TILE]]]
