// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-matmul="matmul-tile-sizes=4,4" \
// RUN: | FileCheck %s

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
// CHECK-LABEL: func @matmul(
// CHECK-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// CHECK-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index

// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// CHECK:      %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1_:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// CHECK:      gml_st.loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1_]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// CHECK-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_2D]])

// CHECK:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]], 0]
// CHECK:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][0, %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]]

// CHECK:      %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]] : [[TY_2D]], [[TY_2D]])
// CHECK-SAME:   outs(%[[OUT_SUB]] : [[TY_2D]])

// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[MATMUL]] into %[[OUT_]]
// CHECK-NEXT: gml_st.yield %[[UPDATE]] : [[TY_2D]]

// -----
