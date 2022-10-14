// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling-softmax="tile-sizes=8,16 distribute=true" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @partial_softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @partial_softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:   %[[C64:.*]] = arith.constant 64
  // CHECK-DAG:   %[[CST:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:      (%[[ARG1:.*]]) = (%[[C0]])
  // CHECK-SAME:      to (%[[C64]]) step (%[[C8]])
  // CHECK-DAG:     %[[SPACE:.*]] = gml_st.space [64, 128]
  // CHECK-DAG:     %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK-DAG:     %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK-DAG:     %[[SPACE_0:.*]] = gml_st.space [64]
  // CHECK-DAG:     %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [8] [1]
  // CHECK-DAG:     %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]][%[[TILE_0]]]
  // CHECK:         %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map1],
  // CHECK-SAME:        iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK:         ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK:           %[[MAXF:.*]] = arith.maxf %[[ARG4]], %[[ARG3]]
  // CHECK:           linalg.yield %[[MAXF]]
  // CHECK:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]][%[[TILE]]]
  // CHECK:         %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map1, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[GENERIC]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_0:.*]]: f32, %[[ARG4_0:.*]]: f32):
  // CHECK:           linalg.yield %[[ARG3_0]]
  // CHECK:         %[[GENERIC_1:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map0, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[MATERIALIZE]], %[[GENERIC_0]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_1:.*]]: f32, %[[ARG4_1:.*]]: f32, %[[ARG5:.*]]: f32):
  // CHECK:           %[[SUBF:.*]] = arith.subf %[[ARG3_1]], %[[ARG4_1]]
  // CHECK:           linalg.yield %[[SUBF]]
  // CHECK:         gml_st.set_yield %[[GENERIC_1]] into %[[INIT_0]][%[[TILE]]]
  // CHECK:       return %[[PARALLEL]]
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.maxf %arg2, %arg1 : f32
    linalg.yield %6 : f32
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
    %6 = arith.subf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<64x128xf32>
  return %5 : tensor<64x128xf32>
}

// -----

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @partial_softmax_fusion
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>, %[[ARG1:.*]]: index
func.func @partial_softmax_fusion(%arg0: tensor<64x128xf32>, %arg1: index)
    -> tensor<8x128xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:   %[[C128:.*]] = arith.constant 128
  // CHECK-DAG:   %[[CST:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK-DAG:   %[[SPACE:.*]] = gml_st.space [64, 128]
  // CHECK-DAG:   %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [%[[C8]], 128] [1, 1]
  // CHECK-DAG:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK-DAG:   %[[SPACE_0:.*]] = gml_st.space [64]
  // CHECK-DAG:   %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [%[[C8]]] [1]
  // CHECK-DAG:   %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]][%[[TILE_0]]]
  // CHECK-DAG:   %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_0]] : tensor<?xf32> to tensor<8xf32>
  // CHECK-DAG:   %[[CAST_0:.*]] = tensor.cast %[[MATERIALIZE]] : tensor<?x128xf32> to tensor<8x128xf32>
  // CHECK:       %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:      indexing_maps = [#map0, #map1],
  // CHECK-SAME:      iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:      ins(%[[CAST_0]] : tensor<8x128xf32>)
  // CHECK-SAME:      outs(%[[CAST]] : tensor<8xf32>)
  // CHECK:       ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK-DAG:     %[[MAXF:.*]] = arith.maxf %[[ARG3]], %[[ARG2]]
  // CHECK:         linalg.yield %[[MAXF]]
  // CHECK-DAG:   %[[TILE_1:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], %[[C0]]] [%[[C8]], %[[C128]]] [1, 1]
  // CHECK-DAG:   %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]][%[[TILE_1]]]
  // CHECK-DAG:   %[[CAST_1:.*]] = tensor.cast %[[MATERIALIZE_1]] : tensor<?x?xf32> to tensor<8x128xf32>
  // CHECK:       %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:      indexing_maps = [#map1, #map0],
  // CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:      ins(%[[GENERIC]] : tensor<8xf32>)
  // CHECK-SAME:      outs(%[[CAST_1]] : tensor<8x128xf32>)
  // CHECK:       ^bb0(%[[ARG2_0:.*]]: f32, %[[ARG3_0:.*]]: f32):
  // CHECK:         linalg.yield %[[ARG2_0]]
  // CHECK:       %[[GENERIC_1:.*]] = linalg.generic
  // CHECK-SAME:      indexing_maps = [#map0, #map0, #map0],
  // CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:      ins(%[[CAST_0]], %[[GENERIC_0]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:      outs(%[[CAST_1]] : tensor<8x128xf32>)
  // CHECK:       ^bb0(%[[ARG2_1:.*]]: f32, %[[ARG3_1:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK-DAG:     %[[SUBF:.*]] = arith.subf %[[ARG2_1]], %[[ARG3_1]]
  // CHECK:         linalg.yield %[[SUBF]]
  // CHECK:       return %[[GENERIC_1]]
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %9 = arith.maxf %arg3, %arg2 : f32
    linalg.yield %9 : f32
  } -> tensor<64xf32>
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    linalg.yield %arg2 : f32
  } -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %9 = arith.subf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  } -> tensor<64x128xf32>
  %6 = gml_st.space [64, 128] : !gml_st.tile<64x128>
  %7 = gml_st.tile %6 [%arg1, 0] [8, 128] [1, 1]
      : !gml_st.tile<64x128> to !gml_st.tile<8x128>
  %8 = gml_st.materialize %5[%7]
      : tensor<64x128xf32>[!gml_st.tile<8x128>] to tensor<8x128xf32>
  return %8 : tensor<8x128xf32>
}

// -----

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:   %[[C64:.*]] = arith.constant 64
  // CHECK-DAG:   %[[C128:.*]] = arith.constant 128
  // CHECK-DAG:   %[[CST:.*]] = arith.constant -0.000000e+00
  // CHECK-DAG:   %[[CST_0:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK-DAG:   %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:      (%[[ARG1:.*]]) = (%[[C0]])
  // CHECK-SAME:      to (%[[C64]]) step (%[[C8]])
  // CHECK:         %[[SPACE:.*]] = gml_st.space [64, 128]
  // CHECK:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [%[[C8]], 128] [1, 1]
  // CHECK:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:         %[[SPACE_0:.*]] = gml_st.space [64]
  // CHECK:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [%[[C8]]] [1]
  // CHECK:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]][%[[TILE_0]]]
  // CHECK:         %[[CAST:.*]] = tensor.cast %[[MATERIALIZE_0]] : tensor<?xf32> to tensor<8xf32>
  // CHECK:         %[[CAST_0:.*]] = tensor.cast %[[MATERIALIZE]] : tensor<?x128xf32> to tensor<8x128xf32>
  // CHECK:         %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map1],
  // CHECK-SAME:        iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:        ins(%[[CAST_0]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[CAST]] : tensor<8xf32>)
  // CHECK:         ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK:           %[[MAXF:.*]] = arith.maxf %[[ARG4]], %[[ARG3]]
  // CHECK:           linalg.yield %[[MAXF]]
  // CHECK:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], %[[C0]]] [%[[C8]], %[[C128]]] [1, 1]
  // CHECK:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]][%[[TILE_1]]]
  // CHECK:         %[[CAST_1:.*]] = tensor.cast %[[MATERIALIZE_1]] : tensor<?x?xf32> to tensor<8x128xf32>
  // CHECK:         %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map1, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[GENERIC]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[CAST_1]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_0:.*]]: f32, %[[ARG4_0:.*]]: f32):
  // CHECK:           linalg.yield %[[ARG3_0]]
  // CHECK:         %[[GENERIC_1:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map0, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[CAST_0]], %[[GENERIC_0]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[CAST_1]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_1:.*]]: f32, %[[ARG4_1:.*]]: f32, %[[ARG5:.*]]: f32):
  // CHECK:           %[[SUBF:.*]] = arith.subf %[[ARG3_1]], %[[ARG4_1]]
  // CHECK:           linalg.yield %[[SUBF]]
  // CHECK:         %[[TILE_2:.*]] = gml_st.tile %[[SPACE]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[INIT_0]][%[[TILE_2]]]
  // CHECK:         %[[GENERIC_2:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[GENERIC_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_2]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_2:.*]]: f32, %[[ARG4_2:.*]]: f32):
  // CHECK:           %[[EXP:.*]] = math.exp %[[ARG3_2]]
  // CHECK:           linalg.yield %[[EXP]]
  // CHECK:         %[[TILE_3:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG1]]] [8] [1]
  // CHECK:         %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[FILL_0]][%[[TILE_3]]]
  // CHECK:         %[[GENERIC_3:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map1],
  // CHECK-SAME:        iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:        ins(%[[GENERIC_2]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_3]] : tensor<8xf32>)
  // CHECK:         ^bb0(%[[ARG3_3:.*]]: f32, %[[ARG4_3:.*]]: f32):
  // CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG4_3]], %[[ARG3_3]]
  // CHECK:           linalg.yield %[[ADDF]]
  // CHECK:         %[[GENERIC_4:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map1, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[GENERIC_3]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_2]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_4:.*]]: f32, %[[ARG4_4:.*]]: f32):
  // CHECK:           linalg.yield %[[ARG3_4]]
  // CHECK:         %[[GENERIC_5:.*]] = linalg.generic
  // CHECK-SAME:        indexing_maps = [#map0, #map0, #map0],
  // CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:        ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_2]] : tensor<8x128xf32>)
  // CHECK:         ^bb0(%[[ARG3_5:.*]]: f32, %[[ARG4_5:.*]]: f32, %[[ARG5_0:.*]]: f32):
  // CHECK:           %[[DIVF:.*]] = arith.divf %[[ARG3_5]], %[[ARG4_5]]
  // CHECK:           linalg.yield %[[DIVF]]
  // CHECK:         gml_st.set_yield %[[GENERIC_5]] into %[[INIT_0]][%[[TILE_2]]]
  // CHECK:       return %[[PARALLEL]]
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
