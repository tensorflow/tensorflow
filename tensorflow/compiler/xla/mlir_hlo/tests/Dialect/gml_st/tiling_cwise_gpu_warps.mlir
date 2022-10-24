// RUN: mlir-hlo-opt %s --split-input-file --gml-tiling-cwise-gpu-warps | \
// RUN: FileCheck %s

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

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @bcast_cwise
// CHECK-SAME:  %[[ARG0:.*]]: tensor<1x4096xf32>, %[[ARG1:.*]]: tensor<f32>
func.func @bcast_cwise(%arg0: tensor<1x4096xf32>, %arg1: tensor<f32>)
    -> tensor<4096xf32> {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C32:.*]] = arith.constant 32
  // CHECK:      %[[C4096:.*]] = arith.constant 4096
  // CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<4096xf32>
  // CHECK:      %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:     indexing_maps = [#map0, #map1]
  // CHECK-SAME:     iterator_types = ["parallel"]
  // CHECK-SAME:     ins(%[[ARG1]] : tensor<f32>)
  // CHECK-SAME:     outs(%[[EMPTY]] : tensor<4096xf32>)
  // CHECK:      ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK:        linalg.yield %[[IN]]
  // CHECK:      %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1{{\]\]}}
  // CHECK:      %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:     indexing_maps = [#map1, #map1, #map1]
  // CHECK-SAME:     iterator_types = ["parallel"]
  // CHECK-SAME:     ins(%[[COLLAPSED]], %[[GENERIC]] : tensor<4096xf32>, tensor<4096xf32>)
  // CHECK-SAME:     outs(%[[EMPTY]] : tensor<4096xf32>) {
  // CHECK:      ^bb0(%[[IN_0:.*]]: f32, %[[IN_0_0:.*]]: f32, %[[OUT_0:.*]]: f32):
  // CHECK:        %[[SUBF:.*]] = arith.subf %[[IN_0]], %[[IN_0_0]]
  // CHECK:        linalg.yield %[[SUBF]] : f32
  // CHECK:      %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[ARG2:.*]]) = (%[[C0]])
  // CHECK-SAME:     to (%[[C32]])
  // CHECK-SAME:     step (%[[C1]])
  // CHECK-SAME:     distribution ("warp")
  // CHECK:        %[[SUBI:.*]] = arith.subi %[[C4096]], %[[ARG2]]
  // CHECK:        %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]]
  // CHECK:        %[[TILE:.*]] = gml_st.tile [%[[ARG2]]] [%[[DIVUI]]] [32]
  // CHECK:        %[[MATERIALIZE:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE]]]
  // CHECK:        %[[FOR:.*]] = gml_st.for
  // CHECK-SAME:       (%[[ARG3:.*]]) = (%[[C0]])
  // CHECK-SAME:       to (%[[DIVUI]]) step (%[[C1]])
  // CHECK-SAME:       outs (%[[ARG4:.*]] = %[[MATERIALIZE]]: tensor<?xf32>)
  // CHECK:          %[[MULI:.*]] = arith.muli %[[ARG3]], %[[C32]]
  // CHECK:          %[[TILE_0:.*]] = gml_st.tile [%[[MULI]]] [1] [1]
  // CHECK:          %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[GENERIC_0]][%[[TILE_0]]]
  // CHECK:          %[[EXP:.*]] = math.exp %[[MATERIALIZE_0]]
  // CHECK:          %[[TILE_1:.*]] = gml_st.tile [%[[ARG3]]] [1] [1]
  // CHECK:          gml_st.set_yield %[[EXP]] into %[[ARG4]][%[[TILE_1]]]
  // CHECK:        gml_st.set_yield %[[FOR]] into %[[MATERIALIZE]][%[[TILE]]]
  // CHECK:      return %[[PARALLEL]]
  %0 = tensor.empty() : tensor<4096xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>)
      outs(%0 : tensor<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4096xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]]
      : tensor<1x4096xf32> into tensor<4096xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%collapsed, %1 : tensor<4096xf32>, tensor<4096xf32>)
      outs(%0 : tensor<4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.subf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<4096xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1],
      iterator_types = ["parallel"]} ins(%2 : tensor<4096xf32>)
      outs(%0 : tensor<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = math.exp %in : f32
    linalg.yield %4 : f32
  } -> tensor<4096xf32>
  return %3 : tensor<4096xf32>
}
