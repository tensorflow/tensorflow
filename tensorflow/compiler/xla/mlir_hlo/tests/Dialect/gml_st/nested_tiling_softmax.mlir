// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling-softmax="tile-sizes=8,16 distribute=true" \
// RUN:     --canonicalize --cse \
// RUN:     --gml-tiling-softmax="tile-sizes=1,1 distribute=true" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK: #map{{[0-9]*}} = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map{{[0-9]*}} = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
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
// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[C8:.*]] = arith.constant 8
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-DAG:  %[[C64:.*]] = arith.constant 64
// CHECK-DAG:  %[[CST:.*]] = arith.constant -0.000000e+00
// CHECK-DAG:  %[[CST_0:.*]] = arith.constant 0xFF800000
// CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32)
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)
// CHECK:      %[[EMPTY_0:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:      %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32)
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)

// CHECK:      %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C8]])
// CHECK-NEXT:   %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [8, 128] [1, 1]
// CHECK-NEXT:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK-NEXT:   %[[TILE_0:.*]] = gml_st.tile [%[[ARG1]]] [8] [1]
// CHECK-NEXT:   %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]][%[[TILE_0]]]
// CHECK-NEXT:   %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE]]]
// CHECK-NEXT:   %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[FILL_0]][%[[TILE_0]]]

// CHECK:        %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C8]]) step (%[[C1]])
// CHECK-NEXT:     %[[TILE_4:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 128] [1, 1]
// CHECK-NEXT:     %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[MATERIALIZE]][%[[TILE_4]]]
// CHECK-NEXT:     %[[TILE_5:.*]] = gml_st.tile [%[[ARG2]]] [1] [1]
// CHECK-NEXT:     %[[MATERIALIZE_5:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_5]]]

// CHECK-NEXT:     %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP0:map[0-9]*]], #[[MAP1:map[0-9]*]]]
// CHECK-SAME:         iterator_types = ["parallel", "reduction"]
// CHECK-SAME:         ins(%[[MATERIALIZE_4]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_5]] : tensor<1xf32>)

// CHECK:          %[[MATERIALIZE_6:.*]] = gml_st.materialize %[[MATERIALIZE_1]][%[[TILE_4]]]
// CHECK-NEXT:     %[[GENERIC_0:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP1]], #[[MAP0]]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]
// CHECK-SAME:         ins(%[[GENERIC]] : tensor<1xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[GENERIC_1:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]
// CHECK-SAME:         ins(%[[MATERIALIZE_4]], %[[GENERIC_0]] : tensor<1x128xf32>, tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[GENERIC_2:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP0]], #[[MAP0]]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]
// CHECK-SAME:         ins(%[[GENERIC_1]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[MATERIALIZE_8:.*]] = gml_st.materialize %[[MATERIALIZE_3]][%[[TILE_5]]]
// CHECK-NEXT:          %[[GENERIC_3:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:         iterator_types = ["parallel", "reduction"]
// CHECK-SAME:         ins(%[[GENERIC_2]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_8]] : tensor<1xf32>)

// CHECK:          %[[GENERIC_4:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP1]], #[[MAP0]]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]
// CHECK-SAME:         ins(%[[GENERIC_3]] : tensor<1xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[GENERIC_5:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]
// CHECK-SAME:         ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<1x128xf32>, tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)
// CHECK:          gml_st.set_yield %[[GENERIC_5]] into %[[MATERIALIZE_1]][%[[TILE_4]]]
// CHECK:        gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY_0]][%[[TILE]]]
// CHECK:      return %[[PARALLEL]]
