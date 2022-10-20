// RUN: mlir-hlo-opt %s --gml-tiling-cwise-gpu-warps | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @unary_cwise
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4096xf32>, %[[ARG1:.*]]: tensor<f32>
func.func @unary_cwise(%arg0: tensor<4096xf32>, %arg1: tensor<f32>) -> tensor<4096xf32> {
  // CHECK: %[[C0:.*]] = arith.constant 0
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK: %[[C32:.*]] = arith.constant 32
  // CHECK: %[[C4096:.*]] = arith.constant 4096
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4096xf32>
  // CHECK: %[[PLOOP:.*]] = gml_st.parallel (%[[LID:.*]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("warp")
  // CHECK:   %[[SUBI:.*]] = arith.subi %[[C4096]], %[[LID]]
  // CHECK:   %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]]
  // CHECK:   %[[LANE_TILE:.*]] = gml_st.tile [%[[LID]]] [%[[DIVUI]]] [32] : !gml_st.tile<?>
  // CHECK:   %[[LANE_INIT:.*]] = gml_st.materialize %[[EMPTY]][%[[LANE_TILE]]] : tensor<4096xf32>[!gml_st.tile<?>] to tensor<?xf32>
  // CHECK:   %[[SLOOP:.*]] = gml_st.for (%[[I:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]]) outs (%[[AGGR:.*]] = %[[LANE_INIT]]: tensor<?xf32>)
  // CHECK:     %[[MULI:.*]] = arith.muli %[[I]], %[[C32]]
  // CHECK:     %[[ITER_TILE:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:     %[[ITER_ARG:.*]] = gml_st.materialize %[[ARG0]][%[[ITER_TILE]]] : tensor<4096xf32>[!gml_st.tile<1>] to f32
  // CHECK:     %[[EXP:.*]] = math.exp %[[ITER_ARG]]
  // CHECK:     %[[ITER_TILE_IN_LANE_TILE:.*]] = gml_st.tile [%[[I]]] [1] [1] : !gml_st.tile<1>
  // CHECK:     gml_st.set_yield %[[EXP]] into %[[AGGR]][%[[ITER_TILE_IN_LANE_TILE]]] : f32 into tensor<?xf32>[!gml_st.tile<1>]
  // CHECK:   gml_st.set_yield %[[SLOOP]] into %[[LANE_INIT]][%[[LANE_TILE]]] : tensor<?xf32> into tensor<?xf32>[!gml_st.tile<?>]
  // CHECK: return %[[PLOOP]] : tensor<4096xf32>
  %0 = tensor.empty() : tensor<4096xf32>
  %1 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<4096xf32>)
      outs(%0 : tensor<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.exp %in : f32
    linalg.yield %2 : f32
  } -> tensor<4096xf32>
  return %1 : tensor<4096xf32>
}
