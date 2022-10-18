// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling-softmax="tile-sizes=8,16 distribute=true" \
// RUN:     --canonicalize --cse \
// RUN:     --gml-tiling-softmax="tile-sizes=1,1 distribute=true" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C8:.*]] = arith.constant 8
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C64:.*]] = arith.constant 64
  // CHECK:      %[[C128:.*]] = arith.constant 128
  // CHECK:      %[[CST:.*]] = arith.constant -0.000000e+00
  // CHECK:      %[[CST_0:.*]] = arith.constant 0xFF800000
  // CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32)
  // CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)
  // CHECK:      %[[EMPTY_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK:      %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32)
  // CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)
  // CHECK:      %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C8]])
  // CHECK:        %[[SPACE:.*]] = gml_st.space [64, 128]
  // CHECK:        %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [%[[C8]], 128] [1, 1]
  // CHECK:        %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:        %[[SPACE_0:.*]] = gml_st.space [64]
  // CHECK:        %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [%[[C8]]] [1]
  // CHECK:        %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]][%[[TILE_0]]]
  // CHECK:        %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_0]]
  // CHECK:        %[[CAST_1:.*]] = tensor.cast %[[MATERIALIZE]]
  // CHECK:        %[[TILE_1:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], %[[C0]]] [%[[C8]], %[[C128]]] [1, 1]
  // CHECK:        %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE_1]]]
  // CHECK:        %[[CAST_2:.*]] = tensor.cast %[[MATERIALIZE_1]]
  // CHECK:        %[[TILE_2:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:        %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE_2]]]
  // CHECK:        %[[TILE_3:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [8] [1]
  // CHECK:        %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[FILL_0]][%[[TILE_3]]]
  // CHECK:        %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C8]]) step (%[[C1]])
  // CHECK:          %[[SPACE_1:.*]] = gml_st.space [8, 128]
  // CHECK:          %[[TILE_4:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], 0] [%[[C1]], 128] [1, 1]
  // CHECK:          %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[CAST_1]][%[[TILE_4]]]
  // CHECK:          %[[SPACE_2:.*]] = gml_st.space [8]
  // CHECK:          %[[TILE_5:.*]] = gml_st.tile %[[SPACE_2]] [%[[ARG2]]] [%[[C1]]] [1]
  // CHECK:          %[[MATERIALIZE_5:.*]] = gml_st.materialize %[[CAST]][%[[TILE_5]]]
  // CHECK:          %[[CAST_3:.*]] = tensor.cast %[[MATERIALIZE_5]]
  // CHECK:          %[[CAST_4:.*]] = tensor.cast %[[MATERIALIZE_4]]
  // CHECK:          %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map0, #map1]
  // CHECK-SAME:         iterator_types = ["parallel", "reduction"]
  // CHECK-SAME:         ins(%[[CAST_4]] : tensor<1x128xf32>)
  // CHECK-SAME:         outs(%[[CAST_3]] : tensor<1xf32>)
  // CHECK:          %[[TILE_6:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], %[[C0]]] [%[[C1]], %[[C128]]] [1, 1]
  // CHECK:          %[[MATERIALIZE_6:.*]] = gml_st.materialize %[[CAST_2]][%[[TILE_6]]]
  // CHECK:          %[[CAST_5:.*]] = tensor.cast %[[MATERIALIZE_6]]
  // CHECK:          %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map1, #map0]
  // CHECK-SAME:         iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:         ins(%[[GENERIC]] : tensor<1xf32>)
  // CHECK-SAME:         outs(%[[CAST_5]] : tensor<1x128xf32>)
  // CHECK:          %[[GENERIC_1:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map0, #map0, #map0]
  // CHECK-SAME:         iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:         ins(%[[CAST_4]], %[[GENERIC_0]] : tensor<1x128xf32>, tensor<1x128xf32>)
  // CHECK-SAME:         outs(%[[CAST_5]] : tensor<1x128xf32>)
  // CHECK:          %[[TILE_7:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], 0] [1, 128] [1, 1]
  // CHECK:          %[[MATERIALIZE_7:.*]] = gml_st.materialize %[[MATERIALIZE_2]][%[[TILE_7]]]
  // CHECK:          %[[GENERIC_2:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map0, #map0]
  // CHECK-SAME:         iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:         ins(%[[GENERIC_1]] : tensor<1x128xf32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_7]] : tensor<1x128xf32>)
  // CHECK:          %[[TILE_8:.*]] = gml_st.tile %[[SPACE_2]] [%[[ARG2]]] [1] [1]
  // CHECK:          %[[MATERIALIZE_8:.*]] = gml_st.materialize %[[MATERIALIZE_3]][%[[TILE_8]]]
  // CHECK:          %[[GENERIC_3:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map0, #map1]
  // CHECK-SAME:         iterator_types = ["parallel", "reduction"]
  // CHECK-SAME:         ins(%[[GENERIC_2]] : tensor<1x128xf32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_8]] : tensor<1xf32>)
  // CHECK:          %[[GENERIC_4:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map1, #map0]
  // CHECK-SAME:         iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:         ins(%[[GENERIC_3]] : tensor<1xf32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_7]] : tensor<1x128xf32>)
  // CHECK:          %[[GENERIC_5:.*]] = linalg.generic
  // CHECK-SAME:         indexing_maps = [#map0, #map0, #map0
  // CHECK-SAME:         iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:         ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<1x128xf32>, tensor<1x128xf32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_7]] : tensor<1x128xf32>)
  // CHECK:          gml_st.set_yield %[[GENERIC_5]] into %[[MATERIALIZE_2]][%[[TILE_7]]]
  // CHECK:        gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY_0]][%[[TILE_2]]]
  // CHECK:      return %[[PARALLEL]]
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.maxf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.subf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %6 = linalg.generic {indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = math.exp %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %8 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<64x128xf32>)
      outs(%7 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.addf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %10 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%6, %9 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.divf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  return %10 : tensor<64x128xf32>
}
