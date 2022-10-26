// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_static(%arg0: tensor<128x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

// CHECK-LABEL:    func @matmul_static(
// CHECK-SAME:       %[[LHS:.*]]: tensor<128x16xf32>,
// CHECK-SAME:       %[[RHS:.*]]: tensor<16x64xf32>,
// CHECK-SAME:       %[[OUT:.*]]: tensor<128x64xf32>)

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:  %[[C_LHS_COL:.*]] = arith.constant 16 : index
// CHECK-DAG:  %[[C_RHS_COL:.*]] = arith.constant 64 : index
// CHECK-DAG:  %[[C_LHS_ROW:.*]] = arith.constant 128 : index

// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:     to (%[[C_LHS_ROW]], %[[C_RHS_COL]]) step (%[[C8]], %[[C4]])

// CHECK:      %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0] [8, 16]
// CHECK:      %[[LHS_SLICE:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]

// CHECK:      %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]] [16, 4]
// CHECK:      %[[RHS_SLICE:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]

// CHECK:      %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]] [8, 4]
// CHECK:      %[[OUT_SLICE:.*]] = gml_st.materialize %[[OUT]][%[[OUT_TILE]]]

// CHECK:      %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK-SAME:   to (%[[C_LHS_COL]]) step (%[[C2]])
// CHECK-SAME:   outs (%[[OUT_SUB_ARG:.*]] = %[[OUT_SLICE]]:

// CHECK:        %[[LHS_TILE_2:.*]] = gml_st.tile [0, %[[K]]] [8, 2]
// CHECK:        %[[LHS_SLICE_2:.*]] = gml_st.materialize %[[LHS_SLICE]][%[[LHS_TILE_2]]]

// CHECK:        %[[RHS_TILE_2:.*]] = gml_st.tile [%[[K]], 0] [2, 4]
// CHECK:        %[[RHS_SLICE_2:.*]] = gml_st.materialize %[[RHS_SLICE]][%[[RHS_TILE_2]]]

// CHECK:        %[[OUT_TILE_2:.*]] = gml_st.tile [0, 0] [8, 4]
// CHECK:        %[[OUT_SLICE_2:.*]] = gml_st.materialize %[[OUT_SUB_ARG]][%[[OUT_TILE_2]]]

// CHECK:        %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:     ins(%[[LHS_SLICE_2]], %[[RHS_SLICE_2]] :
// CHECK:          outs(%[[OUT_SLICE_2]] :

// CHECK-NEXT:   gml_st.set_yield %[[MATMUL]] into %[[OUT_SUB_ARG]][%[[OUT_TILE_2]]]
// CHECK:      gml_st.set_yield %[[FOR]] into %[[OUT]][%[[OUT_TILE]]]

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

// CHECK-LABEL: func @matmul(
// CHECK-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG:     %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index

// CHECK-DAG:     %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// CHECK-DAG:     %[[DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]
// CHECK:         %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_1]]) : [[TY_2D]]
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// CHECK-DAG:     %[[LHS_ROW:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// CHECK-DAG:     %[[LHS_COL:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]
// CHECK-DAG:     %[[RHS_COL:.*]] = tensor.dim %[[RHS]], %[[C1]] : [[TY_2D]]

// CHECK:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:        to (%[[LHS_ROW]], %[[RHS_COL]]) step (%[[C8]], %[[C4]])

// CHECK:           %[[LHS_TILE:.*]] = gml_st.tile [%[[I]], 0]
// CHECK:           %[[LHS_SLICE:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]

// CHECK:           %[[RHS_TILE:.*]] = gml_st.tile [0, %[[J]]]
// CHECK:           %[[RHS_SLICE:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]

// CHECK:           %[[OUT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
// CHECK:           %[[OUT_SLICE:.*]] = gml_st.materialize %[[FILL]][%[[OUT_TILE]]]

// CHECK:           %[[LHS_SUB_ROW:.*]] = tensor.dim %[[LHS_SLICE]], %[[C0]] : [[TY_2D]]
// CHECK:           %[[LHS_SUB_COL:.*]] = tensor.dim %[[LHS_SLICE]], %[[C1]] : [[TY_2D]]
// CHECK:           %[[RHS_SUB_COL:.*]] = tensor.dim %[[RHS_SLICE]], %[[C1]] : [[TY_2D]]

// CHECK:           %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK-SAME:          to (%[[LHS_SUB_COL]]) step (%[[C2]])
// CHECK-SAME:          outs (%[[OUT_SUB_ARG:.*]] = %[[OUT_SLICE]]: [[TY_2D]])

// CHECK:             %[[LHS_TILE_2:.*]] = gml_st.tile [0, %[[K]]]
// CHECK:             %[[LHS_SLICE_2:.*]] = gml_st.materialize %[[LHS_SLICE]][%[[LHS_TILE_2]]]

// CHECK:             %[[RHS_TILE_2:.*]] = gml_st.tile [%[[K]], 0]
// CHECK:             %[[RHS_SLICE_2:.*]] = gml_st.materialize %[[RHS_SLICE]][%[[RHS_TILE_2]]]

// CHECK:             %[[OUT_TILE_2:.*]] = gml_st.tile [0, 0] [%[[LHS_SUB_ROW]], %[[RHS_SUB_COL]]]
// CHECK:             %[[OUT_SLICE_2:.*]] = gml_st.materialize %[[OUT_SUB_ARG]][%[[OUT_TILE_2]]]

// CHECK:             %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:          ins(%[[LHS_SLICE_2]], %[[RHS_SLICE_2]] : [[TY_2D]], [[TY_2D]])
// CHECK:               outs(%[[OUT_SLICE_2]] : [[TY_2D]])

// CHECK-NEXT:        gml_st.set_yield %[[MATMUL]] into %[[OUT_SUB_ARG]][%[[OUT_TILE_2]]]
// CHECK:           gml_st.set_yield %[[FOR]] into %[[FILL]][%[[OUT_TILE]]]
