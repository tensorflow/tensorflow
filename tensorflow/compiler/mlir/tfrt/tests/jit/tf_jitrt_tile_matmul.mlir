// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-matmul="matmul-tile-sizes=0" \
// RUN: | FileCheck %s --check-prefix=TILE-EMPTY

// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-matmul="matmul-tile-sizes=4" \
// RUN: | FileCheck %s --check-prefix=TILE-1D

// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-matmul="matmul-tile-sizes=4,4" \
// RUN: | FileCheck %s --check-prefix=TILE-2D

// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-matmul="matmul-tile-sizes=4,4,4" \
// RUN: | FileCheck %s --check-prefix=TILE-3D

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// TILE-EMPTY-LABEL: func @matmul(
// TILE-EMPTY-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// TILE-EMPTY-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// TILE-EMPTY-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// TILE-EMPTY-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TILE-EMPTY-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TILE-EMPTY:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// TILE-EMPTY:      %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// TILE-EMPTY:      %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// TILE-EMPTY:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]

// TILE-EMPTY:      %[[MATMUL:.*]] = linalg.matmul
// TILE-EMPTY-SAME:   ins(%[[LHS]], %[[RHS]] : [[TY_2D]], [[TY_2D]])
// TILE-EMPTY-SAME:   outs(%[[FILL]] : [[TY_2D]])

// -----

// TILE-1D-LABEL: func @matmul(
// TILE-1D-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// TILE-1D-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// TILE-1D-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// TILE-1D-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TILE-1D-DAG:  %[[C4:.*]] = arith.constant 4 : index
// TILE-1D-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TILE-1D:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// TILE-1D:      %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// TILE-1D:      %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// TILE-1D:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// TILE-1D:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]

// TILE-1D:      gml_st.loop (%[[K:.*]]) = (%[[C0]])
// TILE-1D-SAME:   to (%[[DIM_0_]]) step (%[[C4]])
// TILE-1D-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// TILE-1D-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// TILE-1D-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_2D]])

// TILE-1D:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][0, %[[K]]]
// TILE-1D:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][%[[K]], 0]
// TILE-1D:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][0, 0]

// TILE-1D:      %[[MATMUL:.*]] = linalg.matmul
// TILE-1D-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]] : [[TY_2D]], [[TY_2D]])
// TILE-1D-SAME:   outs(%[[OUT_SUB]] : [[TY_2D]])

// TILE-1D:      %[[UPDATE:.*]] = tensor.insert_slice %[[MATMUL]] into %[[OUT_]][0, 0]
// TILE-1D-NEXT: gml_st.yield %[[UPDATE]] : [[TY_2D]]

// -----

// TILE-2D-LABEL: func @matmul(
// TILE-2D-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// TILE-2D-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// TILE-2D-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// TILE-2D-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TILE-2D-DAG:  %[[C4:.*]] = arith.constant 4 : index
// TILE-2D-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TILE-2D:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// TILE-2D:      %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// TILE-2D:      %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// TILE-2D:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// TILE-2D:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// TILE-2D:      %[[DIM_1_:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// TILE-2D:      gml_st.loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// TILE-2D-SAME:   to (%[[DIM_0_]], %[[DIM_1_]]) step (%[[C4]], %[[C4]])
// TILE-2D-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// TILE-2D-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// TILE-2D-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_2D]])

// TILE-2D:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]], 0]
// TILE-2D:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][0, %[[J]]]
// TILE-2D:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]]

// TILE-2D:      %[[MATMUL:.*]] = linalg.matmul
// TILE-2D-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]] : [[TY_2D]], [[TY_2D]])
// TILE-2D-SAME:   outs(%[[OUT_SUB]] : [[TY_2D]])

// TILE-2D:      %[[UPDATE:.*]] = tensor.insert_slice %[[MATMUL]] into %[[OUT_]][%[[I]], %[[J]]]
// TILE-2D-NEXT: gml_st.yield %[[UPDATE]] : [[TY_2D]]

// -----

// TILE-3D-LABEL: func @matmul(
// TILE-3D-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// TILE-3D-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// TILE-3D-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// TILE-3D-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TILE-3D-DAG:  %[[C4:.*]] = arith.constant 4 : index
// TILE-3D-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TILE-3D:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// TILE-3D:      %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// TILE-3D:      %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// TILE-3D:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// TILE-3D:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// TILE-3D:      %[[DIM_1_:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// TILE-3D:      gml_st.loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// TILE-3D-SAME:   to (%[[DIM_0_]], %[[DIM_1_]]) step (%[[C4]], %[[C4]])
// TILE-3D-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// TILE-3D-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// TILE-3D-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_2D]])

// TILE-3D:      %[[DIM_2_:.*]] = tensor.dim %[[LHS_]], %[[C1]] : [[TY_2D]]
// TILE-3D:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]], 0]
// TILE-3D:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][0, %[[J]]]
// TILE-3D:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]]

// TILE-3D:      %[[LOOP:.*]] = gml_st.loop (%[[K:.*]]) = (%[[C0]])
// TILE-3D-SAME:   to (%[[DIM_2_]]) step (%[[C4]])
// TILE-3D-SAME:   ins (%[[LHS_0:.*]] = %[[LHS_SUB]]: [[TY_2D]],
// TILE-3D-SAME:        %[[RHS_0:.*]] = %[[RHS_SUB]]: [[TY_2D]])
// TILE-3D-SAME:   outs (%[[OUT_0:.*]] = %[[OUT_SUB]]: [[TY_2D]])

// TILE-3D:      %[[LHS_0_SUB:.*]] = tensor.extract_slice %[[LHS_0]][0, %[[K]]]
// TILE-3D:      %[[RHS_0_SUB:.*]] = tensor.extract_slice %[[RHS_0]][%[[K]], 0]
// TILE-3D:      %[[OUT_0_SUB:.*]] = tensor.extract_slice %[[OUT_0]][0, 0]

// TILE-3D:      %[[MATMUL:.*]] = linalg.matmul
// TILE-3D-SAME:   ins(%[[LHS_0_SUB]], %[[RHS_0_SUB]] : [[TY_2D]], [[TY_2D]])
// TILE-3D-SAME:   outs(%[[OUT_0_SUB]] : [[TY_2D]])

// TILE-3D:      %[[UPDATE_0:.*]] = tensor.insert_slice %[[MATMUL]] into %[[OUT_0]][0, 0]
// TILE-3D-NEXT: gml_st.yield %[[UPDATE_0]] : [[TY_2D]]

// TILE-3D:      %[[UPDATE:.*]] = tensor.insert_slice %[[LOOP]] into %[[OUT_]][%[[I]], %[[J]]]
// TILE-3D-NEXT: gml_st.yield %[[UPDATE]] : [[TY_2D]]

// -----
