// RUN: mlir-hlo-opt %s --split-input-file --gml-tiling-gpu-warp \
// RUN: | FILECHECK_OPTS="" FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func @tile_reduction
func.func @tile_reduction(%arg0 : tensor<1x?xf32>) -> tensor<1xf32> {

  %zero = arith.constant 0.0 : f32
  %result0 = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RESULT0:.*]] = linalg.fill
  %result1 = linalg.fill ins(%zero : f32) outs(%result0 : tensor<1xf32>) -> tensor<1xf32>

  // CHECK:      %[[RDIM:.*]] = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  // CHECK:      %[[PARTIAL0:.*]] = tensor.empty() : tensor<32xf32>
  // CHECK:      %[[PARTIAL1:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[LANE:.*]]) = (%c0) to (%c32) step (%c1)
  // CHECK-SAME:     distribution ("thread")
  // CHECK:        %[[INITVAL:.*]] = linalg.fill
  // CHECK:        %[[PARTVAL:.*]] = gml_st.for
  // CHECK-SAME:       (%[[COL:.*]]) = (%[[LANE]]) to (%[[RDIM]]) step (%c32)
  // CHECK-SAME:       outs (%[[OUTVAL:.*]] = %[[INITVAL]]: tensor<1xf32>)
  // CHECK:          %[[TILE:.*]] = gml_st.tile [0, %arg2] [1, 1] [1, 1]
  // CHECK:          %[[INVAL:.*]] = gml_st.materialize %arg0
  // CHECK:          %[[ACCVAL:.*]] = linalg.generic
  // CHECK-SAME:         ins(%[[INVAL]] : tensor<1x1xf32>)
  // CHECK-SAME:         outs(%[[OUTVAL]] : tensor<1xf32>)
  // CHECK:          gml_st.set_yield %[[ACCVAL]] into %[[OUTVAL]]
  // CHECK:        gml_st.set_yield %[[PARTVAL]]
  // CHECK:      %[[EXPAND:.*]] = tensor.expand_shape %[[PARTIAL1]]
  // CHECK:      %[[RESULT1:.*]] = linalg.generic
  // CHECK-SAME:     ins(%[[EXPAND]] : tensor<1x32xf32>)
  // CHECK-SAME:     outs(%[[RESULT0]] : tensor<1xf32>)
  %result2 = linalg.generic {
    indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<1x?xf32>) outs(%result1 : tensor<1xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %24 = arith.addf %arg4, %arg3 : f32
    linalg.yield %24 : f32
  } -> tensor<1xf32>

  // CHECK: return %[[RESULT1]]
  func.return %result2 : tensor<1xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @unary_cwise
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4096xf32>, %[[ARG1:.*]]: tensor<f32>
func.func @unary_cwise(%arg0: tensor<4096xf32>, %arg1: tensor<f32>)
    -> tensor<4096xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[C32:.*]] = arith.constant 32
  // CHECK-DAG:  %[[C4127:.*]] = arith.constant 4127
  // CHECK-DAG:  %[[EMPTY:.*]] = tensor.empty() : tensor<4096xf32>
  // CHECK:      %[[PLOOP:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[LID:.*]]) = (%[[C0]]) to (%[[C32]])
  // CHECK-SAME:     step (%[[C1]]) distribution ("thread")
  // CHECK:      %[[SUBI:.*]] = arith.subi %[[C4127]], %[[LID]]
  // CHECK:      %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]]
  // CHECK:      %[[LANE_TILE:.*]] = gml_st.tile [%[[LID]]] [%[[DIVUI]]] [32]
  // CHECK:      %[[LANE_INIT:.*]] = gml_st.materialize %[[EMPTY]][%[[LANE_TILE]]]
  // CHECK:      %[[SLOOP:.*]] = gml_st.for
  // CHECK-SAME:     (%[[I:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]])
  // CHECK-SAME:     outs (%[[AGGR:.*]] = %[[LANE_INIT]]: tensor<?xf32>)
  // CHECK:        %[[MULI:.*]] = arith.muli %[[I]], %[[C32]]
  // CHECK:        %[[ITER_TILE:.*]] = gml_st.tile [%[[MULI]]] [1] [1]
  // CHECK:        %[[ITER_ARG:.*]] = gml_st.materialize %[[ARG0]][%[[ITER_TILE]]]
  // CHECK:        %[[EXP:.*]] = math.exp %[[ITER_ARG]]
  // CHECK:        %[[ITER_TILE_IN_LANE_TILE:.*]] = gml_st.tile [%[[I]]] [1] [1] : !gml_st.tile<1>
  // CHECK:        gml_st.set_yield %[[EXP]] into %[[AGGR]][%[[ITER_TILE_IN_LANE_TILE]]]
  // CHECK:      gml_st.set_yield %[[SLOOP]] into %[[EMPTY]][%[[LANE_TILE]]]
  // CHECK:    return %[[PLOOP]]
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
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG:  %[[C4127:.*]] = arith.constant 4127 : index
  // CHECK-DAG:  %[[EMPTY:.*]] = tensor.empty() : tensor<4096xf32>
  // CHECK-DAG:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1{{\]\]}}
  // CHECK:      %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C32]])
  // CHECK-SAME:     step (%[[C1]]) distribution ("thread")
  // CHECK:        %[[SUBI:.*]] = arith.subi %[[C4127]], %[[ARG2]] : index
  // CHECK:        %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]] : index
  // CHECK:        %[[TILE:.*]] = gml_st.tile [%[[ARG2]]] [%[[DIVUI]]] [32] : !gml_st.tile<?>
  // CHECK:        %[[MATERIALIZE:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE]]]
  // CHECK:        %[[FOR:.*]] = gml_st.for
  // CHECK-SAME:       (%[[ARG3:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]])
  // CHECK-SAME:       outs (%[[ARG4:.*]] = %[[MATERIALIZE]]: tensor<?xf32>) {
  // CHECK:          %[[MULI:.*]] = arith.muli %[[ARG3]], %[[C32]] : index
  // CHECK:          %[[TILE_0:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:          %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[COLLAPSED]][%[[TILE_0]]]
  // CHECK:          %[[TILE_1:.*]] = gml_st.tile [] [] [] : !gml_st.tile<>
  // CHECK:          %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_1]]]
  // CHECK:          %[[TILE_2:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:          %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_2]]]
  // CHECK:          %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:         iterator_types = ["parallel"]
  // CHECK-SAME:         ins(%[[MATERIALIZE_1]] : tensor<f32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_2]] : tensor<1xf32>)
  // CHECK:          ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK:            linalg.yield %[[IN]] : f32
  // CHECK:          %[[TILE_3:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:          %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_3]]]
  // CHECK:          %[[GENERIC_0:.*]] = linalg.generic
  // CHECK-SAME:         iterator_types = ["parallel"]
  // CHECK-SAME:         ins(%[[MATERIALIZE_0]], %[[GENERIC]] : tensor<1xf32>, tensor<1xf32>)
  // CHECK-SAME:         outs(%[[MATERIALIZE_3]] : tensor<1xf32>)
  // CHECK:          ^bb0(%[[IN_0:.*]]: f32, %[[IN_0_0:.*]]: f32, %[[OUT_0:.*]]: f32):
  // CHECK:            %[[SUBF:.*]] = arith.subf %[[IN_0]], %[[IN_0_0]] : f32
  // CHECK:            linalg.yield %[[SUBF]] : f32
  // CHECK:          %[[EXTRACTED:.*]] = tensor.extract %[[GENERIC_0]][%[[C0]]] : tensor<1xf32>
  // CHECK:          %[[EXP:.*]] = math.exp %[[EXTRACTED]] : f32
  // CHECK:          %[[TILE_4:.*]] = gml_st.tile [%[[ARG3]]] [1] [1] : !gml_st.tile<1>
  // CHECK:          gml_st.set_yield %[[EXP]] into %[[ARG4]][%[[TILE_4]]]
  // CHECK:        gml_st.set_yield %[[FOR]] into %[[EMPTY]][%[[TILE]]]
  // CHECK:      return %[[PARALLEL]] : tensor<4096xf32>
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

// -----

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// CHECK: #[[MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0) -> ()>
// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2048x4096xf32>
func.func @softmax(%arg0: tensor<2048x4096xf32>) -> tensor<2048x4096xf32> {
  // CHECK: %[[C4127:.*]] = arith.constant 4127 : index
  // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK: %[[CST:.*]] = arith.constant -0.000000e+00 : f32
  // CHECK: %[[CST_0:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2048xf32>
  // CHECK: %[[EMPTY_0:.*]] = tensor.empty() : tensor<2048x4096xf32>
  // CHECK: %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C2048]]) step (%[[C1024]]) {
  // CHECK:   %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [1024, 4096] [1, 1] : !gml_st.tile<1024x4096>
  // CHECK:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE]]] : tensor<2048x4096xf32>[!gml_st.tile<1024x4096>] to tensor<1024x4096xf32>
  // CHECK:   %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C1024]]) step (%[[C1]]) {
  // CHECK:     %[[TILE_0:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
  // CHECK:     %[[ADDI:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
  // CHECK:     %[[TILE_1:.*]] = gml_st.tile [%[[ADDI]], 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
  // CHECK:     %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG0]][%[[TILE_1]]] : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
  // CHECK:     %[[TILE_2:.*]] = gml_st.tile [%[[ADDI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:     %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_2]]] : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:     %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[MATERIALIZE_1]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK:     %[[EMPTY_1:.*]] = tensor.empty() : tensor<32xf32>
  // CHECK:     %[[TILE_3:.*]] = gml_st.tile [0] [1] [1] : !gml_st.tile<1>
  // CHECK:     %[[PARALLEL_1:.*]] = gml_st.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[TILE_4:.*]] = gml_st.tile [%[[ARG3]]] [1] [1] : !gml_st.tile<1>
  // CHECK:       gml_st.set_yield %[[FILL]] into %[[EMPTY_1]][%[[TILE_4]]] : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
  // CHECK:     %[[PARALLEL_2:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[TILE_5:.*]] = gml_st.tile [%[[ARG3]]] [1] [1] : !gml_st.tile<1>
  // CHECK:       %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[PARALLEL_1]][%[[TILE_5]]] : tensor<32xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:       %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5:.*]] = %[[MATERIALIZE_2]]: tensor<1xf32>) {
  // CHECK:         %[[TILE_6:.*]] = gml_st.tile [0, %[[ARG4]]] [1, 1] [1, 1] : !gml_st.tile<1x1>
  // CHECK:         %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_6]]] : tensor<1x4096xf32>[!gml_st.tile<1x1>] to f32
  // CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[ARG5]][%[[C0]]] : tensor<1xf32>
  // CHECK:         %[[MAXF:.*]] = arith.maxf %[[EXTRACTED]], %[[MATERIALIZE_3]] : f32
  // CHECK:         gml_st.set_yield %[[MAXF]] into %[[ARG5]][%[[TILE_3]]] : f32 into tensor<1xf32>[!gml_st.tile<1>]
  // CHECK:       gml_st.set_yield %[[FOR]] into %[[PARALLEL_1]][%[[TILE_5]]] : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
  // CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[FILL]] [] : tensor<1xf32> into tensor<f32>
  // CHECK:     %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["reduction"]} ins(%[[PARALLEL_2]] : tensor<32xf32>) outs(%[[COLLAPSED]] : tensor<f32>) {
  // CHECK:     ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK:       %[[MAXF_0:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
  // CHECK:       linalg.yield %[[MAXF_0]] : f32
  // CHECK:     %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE_1]]] : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
  // CHECK:     %[[COLLAPSED_1:.*]] = tensor.collapse_shape %[[MATERIALIZE_4]] {{\[\[}}0, 1{{\]\]}} : tensor<1x4096xf32> into tensor<4096xf32>
  // CHECK:     %[[COLLAPSED_2:.*]] = tensor.collapse_shape %[[MATERIALIZE_0]] {{\[\[}}0, 1{{\]\]}} : tensor<1x4096xf32> into tensor<4096xf32>
  // CHECK:     %[[PARALLEL_3:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[SUBI:.*]] = arith.subi %[[C4127]], %[[ARG3]] : index
  // CHECK:       %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]] : index
  // CHECK:       %[[TILE_7:.*]] = gml_st.tile [%[[ARG3]]] [%[[DIVUI]]] [32] : !gml_st.tile<?>
  // CHECK:       %[[MATERIALIZE_5:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_7]]] : tensor<4096xf32>[!gml_st.tile<?>] to tensor<?xf32>
  // CHECK:       %[[FOR_0:.*]] = gml_st.for (%[[ARG4_0:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]]) outs (%[[ARG5_0:.*]] = %[[MATERIALIZE_5]]: tensor<?xf32>) {
  // CHECK:         %[[MULI:.*]] = arith.muli %[[ARG4_0]], %[[C32]] : index
  // CHECK:         %[[TILE_8:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_6:.*]] = gml_st.materialize %[[COLLAPSED_2]][%[[TILE_8]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[TILE_9:.*]] = gml_st.tile [] [] [] : !gml_st.tile<>
  // CHECK:         %[[MATERIALIZE_7:.*]] = gml_st.materialize %[[GENERIC]][%[[TILE_9]]] : tensor<f32>[!gml_st.tile<>] to tensor<f32>
  // CHECK:         %[[TILE_10:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_8:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_10]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[MATERIALIZE_7]] : tensor<f32>) outs(%[[MATERIALIZE_8]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_0:.*]]: f32, %[[OUT_0:.*]]: f32):
  // CHECK:           linalg.yield %[[IN_0]] : f32
  // CHECK:         %[[TILE_11:.*]] = gml_st.tile [%[[MULI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_9:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_11]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[MATERIALIZE_6]], %[[GENERIC_0]] : tensor<1xf32>, tensor<1xf32>) outs(%[[MATERIALIZE_9]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_1:.*]]: f32, %[[IN_5:.*]]: f32, %[[OUT_1:.*]]: f32):
  // CHECK:           %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_5]] : f32
  // CHECK:           linalg.yield %[[SUBF]] : f32
  // CHECK:         %[[EXTRACTED_0:.*]] = tensor.extract %[[GENERIC_1]][%[[C0]]] : tensor<1xf32>
  // CHECK:         %[[EXP:.*]] = math.exp %[[EXTRACTED_0]] : f32
  // CHECK:         %[[TILE_12:.*]] = gml_st.tile [%[[ARG4_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         gml_st.set_yield %[[EXP]] into %[[ARG5_0]][%[[TILE_12]]] : f32 into tensor<?xf32>[!gml_st.tile<1>]
  // CHECK:       gml_st.set_yield %[[FOR_0]] into %[[COLLAPSED_1]][%[[TILE_7]]] : tensor<?xf32> into tensor<4096xf32>[!gml_st.tile<?>]
  // CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[PARALLEL_3]] {{\[\[}}0, 1{{\]\]}} : tensor<4096xf32> into tensor<1x4096xf32>
  // CHECK:     %[[TILE_13:.*]] = gml_st.tile [%[[ADDI]]] [1] [1] : !gml_st.tile<1>
  // CHECK:     %[[MATERIALIZE_10:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_13]]] : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:     %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_10]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK:     %[[PARALLEL_4:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[TILE_14:.*]] = gml_st.tile [%[[ARG3]]] [1] [1] : !gml_st.tile<1>
  // CHECK:       gml_st.set_yield %[[FILL_0]] into %[[EMPTY_1]][%[[TILE_14]]] : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
  // CHECK:     %[[PARALLEL_5:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[TILE_15:.*]] = gml_st.tile [%[[ARG3]]] [1] [1] : !gml_st.tile<1>
  // CHECK:       %[[MATERIALIZE_11:.*]] = gml_st.materialize %[[PARALLEL_4]][%[[TILE_15]]] : tensor<32xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:       %[[FOR_1:.*]] = gml_st.for (%[[ARG4_1:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5_1:.*]] = %[[MATERIALIZE_11]]: tensor<1xf32>) {
  // CHECK:         %[[TILE_16:.*]] = gml_st.tile [0, %[[ARG4_1]]] [1, 1] [1, 1] : !gml_st.tile<1x1>
  // CHECK:         %[[MATERIALIZE_12:.*]] = gml_st.materialize %[[EXPANDED]][%[[TILE_16]]] : tensor<1x4096xf32>[!gml_st.tile<1x1>] to f32
  // CHECK:         %[[EXTRACTED_1:.*]] = tensor.extract %[[ARG5_1]][%[[C0]]] : tensor<1xf32>
  // CHECK:         %[[ADDF:.*]] = arith.addf %[[EXTRACTED_1]], %[[MATERIALIZE_12]] : f32
  // CHECK:         gml_st.set_yield %[[ADDF]] into %[[ARG5_1]][%[[TILE_3]]] : f32 into tensor<1xf32>[!gml_st.tile<1>]
  // CHECK:       } : tensor<1xf32>
  // CHECK:       gml_st.set_yield %[[FOR_1]] into %[[PARALLEL_4]][%[[TILE_15]]] : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
  // CHECK:     %[[COLLAPSED_3:.*]] = tensor.collapse_shape %[[FILL_0]] [] : tensor<1xf32> into tensor<f32>
  // CHECK:     %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["reduction"]} ins(%[[PARALLEL_5]] : tensor<32xf32>) outs(%[[COLLAPSED_3]] : tensor<f32>) {
  // CHECK:     ^bb0(%[[IN_2:.*]]: f32, %[[OUT_2:.*]]: f32):
  // CHECK:       %[[ADDF_0:.*]] = arith.addf %[[OUT_2]], %[[IN_2]] : f32
  // CHECK:       linalg.yield %[[ADDF_0]] : f32
  // CHECK:     %[[PARALLEL_6:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread") {
  // CHECK:       %[[SUBI_0:.*]] = arith.subi %[[C4127]], %[[ARG3]] : index
  // CHECK:       %[[DIVUI_0:.*]] = arith.divui %[[SUBI_0]], %[[C32]] : index
  // CHECK:       %[[TILE_17:.*]] = gml_st.tile [%[[ARG3]]] [%[[DIVUI_0]]] [32] : !gml_st.tile<?>
  // CHECK:       %[[MATERIALIZE_13:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_17]]] : tensor<4096xf32>[!gml_st.tile<?>] to tensor<?xf32>
  // CHECK:       %[[FOR_2:.*]] = gml_st.for (%[[ARG4_2:.*]]) = (%[[C0]]) to (%[[DIVUI_0]]) step (%[[C1]]) outs (%[[ARG5_2:.*]] = %[[MATERIALIZE_13]]: tensor<?xf32>) {
  // CHECK:         %[[MULI_0:.*]] = arith.muli %[[ARG4_2]], %[[C32]] : index
  // CHECK:         %[[TILE_18:.*]] = gml_st.tile [%[[MULI_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_14:.*]] = gml_st.materialize %[[COLLAPSED_2]][%[[TILE_18]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[TILE_19:.*]] = gml_st.tile [] [] [] : !gml_st.tile<>
  // CHECK:         %[[MATERIALIZE_15:.*]] = gml_st.materialize %[[GENERIC]][%[[TILE_19]]] : tensor<f32>[!gml_st.tile<>] to tensor<f32>
  // CHECK:         %[[TILE_20:.*]] = gml_st.tile [%[[MULI_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_16:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_20]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[MATERIALIZE_15]] : tensor<f32>) outs(%[[MATERIALIZE_16]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_3:.*]]: f32, %[[OUT_3:.*]]: f32):
  // CHECK:           linalg.yield %[[IN_3]] : f32
  // CHECK:         %[[TILE_21:.*]] = gml_st.tile [%[[MULI_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_17:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_21]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[MATERIALIZE_14]], %[[GENERIC_3]] : tensor<1xf32>, tensor<1xf32>) outs(%[[MATERIALIZE_17]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_4:.*]]: f32, %[[IN_6:.*]]: f32, %[[OUT_4:.*]]: f32):
  // CHECK:           %[[SUBF_0:.*]] = arith.subf %[[IN_4]], %[[IN_6]] : f32
  // CHECK:           linalg.yield %[[SUBF_0]] : f32
  // CHECK:         %[[TILE_22:.*]] = gml_st.tile [%[[MULI_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_18:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_22]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_4]] : tensor<1xf32>) outs(%[[MATERIALIZE_18]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_7:.*]]: f32, %[[OUT_5:.*]]: f32):
  // CHECK:           %[[EXP_0:.*]] = math.exp %[[IN_7]] : f32
  // CHECK:           linalg.yield %[[EXP_0]] : f32
  // CHECK:         %[[EXTRACTED_2:.*]] = tensor.extract %[[GENERIC_5]][%[[C0]]] : tensor<1xf32>
  // CHECK:         %[[TILE_23:.*]] = gml_st.tile [] [] [] : !gml_st.tile<>
  // CHECK:         %[[MATERIALIZE_19:.*]] = gml_st.materialize %[[GENERIC_2]][%[[TILE_23]]] : tensor<f32>[!gml_st.tile<>] to tensor<f32>
  // CHECK:         %[[TILE_24:.*]] = gml_st.tile [%[[MULI_0]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         %[[MATERIALIZE_20:.*]] = gml_st.materialize %[[COLLAPSED_1]][%[[TILE_24]]] : tensor<4096xf32>[!gml_st.tile<1>] to tensor<1xf32>
  // CHECK:         %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP0]]], iterator_types = ["parallel"]} ins(%[[MATERIALIZE_19]] : tensor<f32>) outs(%[[MATERIALIZE_20]] : tensor<1xf32>) {
  // CHECK:         ^bb0(%[[IN_8:.*]]: f32, %[[OUT_6:.*]]: f32):
  // CHECK:           linalg.yield %[[IN_8]] : f32
  // CHECK:         %[[EXTRACTED_5:.*]] = tensor.extract %[[GENERIC_6]][%[[C0]]] : tensor<1xf32>
  // CHECK:         %[[DIVF:.*]] = arith.divf %[[EXTRACTED_2]], %[[EXTRACTED_5]] : f32
  // CHECK:         %[[TILE_25:.*]] = gml_st.tile [%[[ARG4_2]]] [1] [1] : !gml_st.tile<1>
  // CHECK:         gml_st.set_yield %[[DIVF]] into %[[ARG5_2]][%[[TILE_25]]] : f32 into tensor<?xf32>[!gml_st.tile<1>]
  // CHECK:       gml_st.set_yield %[[FOR_2]] into %[[COLLAPSED_1]][%[[TILE_17]]] : tensor<?xf32> into tensor<4096xf32>[!gml_st.tile<?>]
  // CHECK:     %[[EXPANDED_4:.*]] = tensor.expand_shape %[[PARALLEL_6]] {{\[\[}}0, 1{{\]\]}} : tensor<4096xf32> into tensor<1x4096xf32>
  // CHECK:     gml_st.set_yield %[[EXPANDED_4]] into %[[MATERIALIZE]][%[[TILE_0]]] : tensor<1x4096xf32> into tensor<1024x4096xf32>[!gml_st.tile<1x4096>]
  // CHECK:   gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY_0]][%[[TILE]]] : tensor<1024x4096xf32> into tensor<2048x4096xf32>[!gml_st.tile<1024x4096>]
  // CHECK: return %[[PARALLEL]] : tensor<2048x4096xf32>
  %c4096 = arith.constant 4096 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2048xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<2048xf32>)
      -> tensor<2048xf32>
  %2 = tensor.empty() : tensor<2048x4096xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048xf32>)
      -> tensor<2048xf32>
  %4 = gml_st.parallel (%arg1) = (%c0) to (%c2048) step (%c1024) {
    %5 = gml_st.tile [%arg1, 0] [1024, 4096] [1, 1] : !gml_st.tile<1024x4096>
    %6 = gml_st.materialize %2[%5]
        : tensor<2048x4096xf32>[!gml_st.tile<1024x4096>]
        to tensor<1024x4096xf32>
    %7 = gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c1) {
      %8 = gml_st.tile [%arg2, 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
      %9 = arith.addi %arg1, %arg2 : index
      %10 = gml_st.tile [%9, 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
      %11 = gml_st.materialize %arg0[%10]
          : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
      %12 = gml_st.tile [%9] [1] [1] : !gml_st.tile<1>
      %13 = gml_st.materialize %1[%12]
          : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
      %14 = tensor.empty() : tensor<32xf32>
      %15 = gml_st.tile [0] [1] [1] : !gml_st.tile<1>
      %16 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1)
          distribution ("thread") {
        %29 = gml_st.tile [%arg3] [1] [1] : !gml_st.tile<1>
        gml_st.set_yield %13 into %14[%29]
            : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
      } : tensor<32xf32>
      %17 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1)
      distribution ("thread") {
        %29 = gml_st.tile [%arg3] [1] [1] : !gml_st.tile<1>
        %30 = gml_st.materialize %16[%29]
            : tensor<32xf32>[!gml_st.tile<1>] to tensor<1xf32>
        %31 = gml_st.for (%arg4) = (%arg3) to (%c4096) step (%c32)
            outs (%arg5 = %30: tensor<1xf32>) {
          %32 = gml_st.tile [0, %arg4] [1, 1] [1, 1] : !gml_st.tile<1x1>
          %33 = gml_st.materialize %11[%32]
              : tensor<1x4096xf32>[!gml_st.tile<1x1>] to f32
          %extracted = tensor.extract %arg5[%c0] : tensor<1xf32>
          %34 = arith.maxf %extracted, %33 : f32
          gml_st.set_yield %34 into %arg5[%15]
              : f32 into tensor<1xf32>[!gml_st.tile<1>]
        } : tensor<1xf32>
        gml_st.set_yield %31 into %16[%29]
            : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
      } : tensor<32xf32>
      %collapsed = tensor.collapse_shape %13 [] : tensor<1xf32> into tensor<f32>
      %18 = linalg.generic {indexing_maps = [#map0, #map1],
          iterator_types = ["reduction"]} ins(%17 : tensor<32xf32>)
          outs(%collapsed : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %29 = arith.maxf %out, %in : f32
        linalg.yield %29 : f32
      } -> tensor<f32>
      %19 = gml_st.materialize %2[%10]
          : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
      %collapsed_1 = tensor.collapse_shape %19 [[0, 1]]
          : tensor<1x4096xf32> into tensor<4096xf32>
      %20 = linalg.generic {indexing_maps = [#map1, #map0],
          iterator_types = ["parallel"]} ins(%18 : tensor<f32>)
          outs(%collapsed_1 : tensor<4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4096xf32>
      %collapsed_2 = tensor.collapse_shape %11 [[0, 1]]
          : tensor<1x4096xf32> into tensor<4096xf32>
      %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
          iterator_types = ["parallel"]}
          ins(%collapsed_2, %20 : tensor<4096xf32>, tensor<4096xf32>)
          outs(%collapsed_1 : tensor<4096xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %29 = arith.subf %in, %in_5 : f32
        linalg.yield %29 : f32
      } -> tensor<4096xf32>
      %22 = linalg.generic {indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} ins(%21 : tensor<4096xf32>)
          outs(%collapsed_1 : tensor<4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        %29 = math.exp %in : f32
        linalg.yield %29 : f32
      } -> tensor<4096xf32>
      %expanded = tensor.expand_shape %22 [[0, 1]]
          : tensor<4096xf32> into tensor<1x4096xf32>
      %23 = gml_st.materialize %3[%12]
          : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
      %24 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1)
          distribution ("thread") {
        %29 = gml_st.tile [%arg3] [1] [1] : !gml_st.tile<1>
        gml_st.set_yield %23 into %14[%29]
            : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
      } : tensor<32xf32>
      %25 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1)
          distribution ("thread") {
        %29 = gml_st.tile [%arg3] [1] [1] : !gml_st.tile<1>
        %30 = gml_st.materialize %24[%29]
            : tensor<32xf32>[!gml_st.tile<1>] to tensor<1xf32>
        %31 = gml_st.for (%arg4) = (%arg3) to (%c4096) step (%c32)
            outs (%arg5 = %30: tensor<1xf32>) {
          %32 = gml_st.tile [0, %arg4] [1, 1] [1, 1] : !gml_st.tile<1x1>
          %33 = gml_st.materialize %expanded[%32]
              : tensor<1x4096xf32>[!gml_st.tile<1x1>] to f32
          %extracted = tensor.extract %arg5[%c0] : tensor<1xf32>
          %34 = arith.addf %extracted, %33 : f32
          gml_st.set_yield %34 into %arg5[%15]
              : f32 into tensor<1xf32>[!gml_st.tile<1>]
        } : tensor<1xf32>
        gml_st.set_yield %31 into %24[%29]
            : tensor<1xf32> into tensor<32xf32>[!gml_st.tile<1>]
      } : tensor<32xf32>
      %collapsed_3 = tensor.collapse_shape %23 []
          : tensor<1xf32> into tensor<f32>
      %26 = linalg.generic {indexing_maps = [#map0, #map1],
          iterator_types = ["reduction"]} ins(%25 : tensor<32xf32>)
          outs(%collapsed_3 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %29 = arith.addf %out, %in : f32
        linalg.yield %29 : f32
      } -> tensor<f32>
      %27 = linalg.generic {indexing_maps = [#map1, #map0],
          iterator_types = ["parallel"]} ins(%26 : tensor<f32>)
          outs(%collapsed_1 : tensor<4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4096xf32>
      %28 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
          iterator_types = ["parallel"]}
          ins(%22, %27 : tensor<4096xf32>, tensor<4096xf32>)
          outs(%collapsed_1 : tensor<4096xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %29 = arith.divf %in, %in_5 : f32
        linalg.yield %29 : f32
      } -> tensor<4096xf32>
      %expanded_4 = tensor.expand_shape %28 [[0, 1]]
          : tensor<4096xf32> into tensor<1x4096xf32>
      gml_st.set_yield %expanded_4 into %6[%8]
          : tensor<1x4096xf32> into tensor<1024x4096xf32>[!gml_st.tile<1x4096>]
    } : tensor<1024x4096xf32>
    gml_st.set_yield %7 into %2[%5] : tensor<1024x4096xf32> into
        tensor<2048x4096xf32>[!gml_st.tile<1024x4096>]
  } : tensor<2048x4096xf32>
  return %4 : tensor<2048x4096xf32>
}
