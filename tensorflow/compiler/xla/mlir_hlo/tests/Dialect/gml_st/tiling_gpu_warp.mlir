// RUN: mlir-hlo-opt %s --split-input-file --gml-tiling-gpu-warp | \
// RUN: FileCheck %s --dump-input=always

// CHECK-LABEL: @tiling_warp_level_reduction
// CHECK-SAME:  %[[ARG0:.*]]: tensor<7x13xf32>
func.func @tiling_warp_level_reduction(%arg0: tensor<7x13xf32>)
    -> tensor<7xf32> {
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[C13:.*]] = arith.constant 13 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST:.*]] = arith.constant 0xFF800000 : f32
  //     CHECK: %[[EMPTY:.*]] = tensor.empty()
  //     CHECK: %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C1024]]) step (%[[C1]]) distribution ("warp")
  //     CHECK:   %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [1, 13] [1, 1]
  //     CHECK:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  //     CHECK:   %[[TILE_0:.*]] = gml_st.tile [%[[ARG1]]] [1] [1]
  //     CHECK:   %[[TILE_1:.*]] = gml_st.tile [%[[ARG1]]] [1] [1]
  //     CHECK:   %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_1]]]
  //     CHECK:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_0]] : tensor<1xf32>)
  //     CHECK:   %[[EMPTY_0:.*]] = tensor.empty()
  //     CHECK:   %[[TILE_2:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:   %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_2]]]
  //     CHECK:   %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_1]] : tensor<1xf32>)
  //     CHECK:   %[[EXTRACTED:.*]] = tensor.extract %[[FILL_0]][%[[C0]]]
  //     CHECK:   %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C16]]) step (%[[C1]]) distribution ("thread")
  //     CHECK:     %[[TILE_4:.*]] = gml_st.tile [0, %[[ARG2]]] [1, 1] [1, 1]
  //     CHECK:     %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE_4]]]
  //     CHECK:     %[[FILL_1:.*]] = linalg.fill ins(%[[EXTRACTED]] : f32) outs(%[[MATERIALIZE_2]] : tensor<1x1xf32>)
  //     CHECK:     %[[FOR:.*]] = gml_st.for (%[[ARG3:.*]]) = (%[[ARG2]]) to (%[[C13]]) step (%[[C16]]) outs (%[[ARG4:.*]] = %[[FILL_1]]: tensor<1x1xf32>)
  //     CHECK:       %[[TILE_5:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[MATERIALIZE]][%[[TILE_5]]] : tensor<1x13xf32>[!gml_st.tile<1x1>] to f32
  //     CHECK:       %[[TILE_6:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:       %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[ARG4]][%[[TILE_6]]] : tensor<1x1xf32>[!gml_st.tile<1x1>] to f32
  //     CHECK:       %[[MAXF:.*]] = arith.maxf %[[MATERIALIZE_4]], %[[MATERIALIZE_3]] : f32
  //     CHECK:       %[[TILE_6_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[MAXF]] into %[[ARG4]][%[[TILE_6_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:     %[[TILE_3:.*]] = gml_st.tile [0, %[[ARG2]]] [1, 1] [1, 1]
  //     CHECK:     gml_st.set_yield %[[FOR]] into %[[EMPTY_0]][%[[TILE_3]]]
  //     CHECK:   %[[REDUCE:.*]] = linalg.reduce
  //     CHECK:       ins(%[[PARALLEL_0]] : tensor<1x16xf32>)
  //     CHECK:       outs(%[[FILL]] : tensor<1xf32>)
  //     CHECK:       dimensions = [1]
  //     CHECK:       (%[[IN:.*]]: f32, %[[OUT:.*]]: f32) {
  //     CHECK:     %[[MAXF_0:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
  //     CHECK:     linalg.yield %[[MAXF_0]] : f32
  //     CHECK:   gml_st.set_yield %[[REDUCE]] into %[[EMPTY]][%[[TILE_0]]]
  //     CHECK: return %[[PARALLEL]]
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<7xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7xf32>)
      -> tensor<7xf32>
  %2 = gml_st.parallel (%arg1) = (%c0) to (%c1024) step (%c1)
      distribution ("warp") {
    %3 = gml_st.tile [%arg1, 0] [1, 13] [1, 1] : !gml_st.tile<1x13>
    %4 = gml_st.materialize %arg0[%3]
        : tensor<7x13xf32>[!gml_st.tile<1x13>] to tensor<1x13xf32>
    %5 = gml_st.tile [%arg1] [1] [1] : !gml_st.tile<1>
    %6 = gml_st.materialize %1[%5]
        : tensor<7xf32>[!gml_st.tile<1>] to tensor<1xf32>
    %7 = linalg.reduce
        ins(%4 : tensor<1x13xf32>)
        outs(%6 : tensor<1xf32>)
        dimensions = [1]
        (%in: f32, %out: f32) {
          %8 = arith.maxf %out, %in : f32
          linalg.yield %8 : f32
        }
    gml_st.set_yield %7 into %1[%5]
        : tensor<1xf32> into tensor<7xf32>[!gml_st.tile<1>]
  } : tensor<7xf32>
  return %2 : tensor<7xf32>
}

// -----

// CHECK-LABEL: @tiling_warp_level_cwise
// CHECK-SAME:  %[[ARG0:.*]]: tensor<7x13xf32>, %[[ARG1:.*]]: tensor<7x13xf32>
func.func @tiling_warp_level_cwise(%arg0: tensor<7x13xf32>,
    %arg1: tensor<7x13xf32>) -> tensor<7x13xf32> {
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C16:.*]] = arith.constant 16
  // CHECK-DAG:  %[[C1024:.*]] = arith.constant 1024
  // CHECK-DAG:  %[[C28:.*]] = arith.constant 28
  // CHECK-DAG:  %[[EMPTY:.*]] = tensor.empty() : tensor<7x13xf32>
  // CHECK:      %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C1024]])
  // CHECK-SAME:     step (%[[C1]]) distribution ("warp")
  // CHECK:        %[[TILE:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 13] [1, 1]
  // CHECK:        %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:        %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE]]]
  // CHECK:        %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE]]]
  // CHECK:        %[[PARALLEL_0:.*]] = gml_st.parallel
  // CHECK-SAME:       (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C16]]) step (%[[C1]])
  // CHECK-SAME:       distribution ("thread")
  // CHECK:          %[[SUBI:.*]] = arith.subi %[[C28]], %[[ARG3]]
  // CHECK:          %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C16]]
  // CHECK:          %[[TILE_0:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 16]
  // CHECK:          %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[MATERIALIZE_1]][%[[TILE_0]]]
  // CHECK:          %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]]) = (%[[C0]])
  // CHECK-SAME:         to (%[[DIVUI]]) step (%[[C1]])
  // CHECK-SAME:         outs (%[[ARG5:.*]] = %[[MATERIALIZE_2]]: tensor<1x?xf32>)
  // CHECK:            %[[MULI:.*]] = arith.muli %[[ARG4]], %[[C16]] : index
  // CHECK:            %[[ADDI:.*]] = arith.addi %[[ARG3]], %[[MULI]] : index
  // CHECK:            %[[TILE_1:.*]] = gml_st.tile [0, %[[ADDI]]] [1, 1] [1, 1]
  // CHECK:            %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[MATERIALIZE]][%[[TILE_1]]]
  // CHECK:            %[[TILE_1_:.*]] = gml_st.tile [0, %[[ADDI]]] [1, 1] [1, 1]
  // CHECK:            %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_1_]]]
  // CHECK:            %[[SUBF:.*]] = arith.subf %[[MATERIALIZE_3]], %[[MATERIALIZE_4]]
  // CHECK:            %[[TILE_2:.*]] = gml_st.tile [0, %[[ARG4]]] [1, 1] [1, 1]
  // CHECK:            gml_st.set_yield %[[SUBF]] into %[[ARG5]][%[[TILE_2]]]
  // CHECK:          %[[TILE_0:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 16]
  // CHECK:          gml_st.set_yield %[[FOR]] into %[[MATERIALIZE_1]][%[[TILE_0]]]
  // CHECK:        gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY]][%[[TILE]]]
  // CHECK:      return %[[PARALLEL]]
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<7x13xf32>
  %1 = gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c1)
      distribution ("warp") {
    %2 = gml_st.tile [%arg2, 0] [1, 13] [1, 1] : !gml_st.tile<1x13>
    %3 = gml_st.materialize %arg0[%2]
        : tensor<7x13xf32>[!gml_st.tile<1x13>] to tensor<1x13xf32>
    %4 = gml_st.materialize %arg1[%2]
        : tensor<7x13xf32>[!gml_st.tile<1x13>] to tensor<1x13xf32>
    %5 = gml_st.materialize %0[%2]
        : tensor<7x13xf32>[!gml_st.tile<1x13>] to tensor<1x13xf32>
    %6 = linalg.map
        ins(%3, %4 : tensor<1x13xf32>, tensor<1x13xf32>)
        outs(%5 : tensor<1x13xf32>)
        (%in: f32, %in_0: f32) {
          %7 = arith.subf %in, %in_0 : f32
          linalg.yield %7 : f32
        }
    gml_st.set_yield %6 into %0[%2]
        : tensor<1x13xf32> into tensor<7x13xf32>[!gml_st.tile<1x13>]
  } : tensor<7x13xf32>
  return %1 : tensor<7x13xf32>
}

// -----

// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2048x4096xf32>
func.func @softmax(%arg0: tensor<2048x4096xf32>) -> tensor<2048x4096xf32> {
  // CHECK-DAG: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[C4127:.*]] = arith.constant 4127 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[CST:.*]] = arith.constant -0.000000e+00 : f32
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant 0xFF800000 : f32
  //     CHECK: %[[EMPTY:.*]] = tensor.empty()
  //     CHECK: %[[EMPTY_0:.*]] = tensor.empty()
  //     CHECK: %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C2048]]) step (%[[C1024]]) distribution ("block")
  //     CHECK:   %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [1024, 4096] [1, 1]
  //     CHECK:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE]]]
  //     CHECK:   %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C1024]]) step (%[[C1]]) distribution ("warp")
  //     CHECK:     %[[TILE_0:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 4096] [1, 1]
  //     CHECK:     %[[ADDI:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
  //     CHECK:     %[[TILE_1:.*]] = gml_st.tile [%[[ADDI]], 0] [1, 4096] [1, 1]
  //     CHECK:     %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG0]][%[[TILE_1]]]
  //     CHECK:     %[[TILE_2:.*]] = gml_st.tile [%[[ADDI]]] [1] [1]
  //     CHECK:     %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_2]]]
  //     CHECK:     %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[MATERIALIZE_1]] : tensor<1xf32>)
  //     CHECK:     %[[EMPTY_1:.*]] = tensor.empty()
  //     CHECK:     %[[TILE_3:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:     %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[MATERIALIZE_1]][%[[TILE_3]]]
  //     CHECK:     %[[FILL_0:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[MATERIALIZE_2]] : tensor<1xf32>)
  //     CHECK:     %[[EXTRACTED:.*]] = tensor.extract %[[FILL_0]][%[[C0]]]
  //     CHECK:     %[[PARALLEL_1:.*]] = gml_st.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread")
  //     CHECK:       %[[TILE_5:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[EMPTY_1]][%[[TILE_5]]]
  //     CHECK:       %[[FILL_1:.*]] = linalg.fill ins(%[[EXTRACTED]] : f32) outs(%[[MATERIALIZE_3]] : tensor<1x1xf32>)
  //     CHECK:       %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5:.*]] = %[[FILL_1]]: tensor<1x1xf32>)
  //     CHECK:         %[[TILE_6:.*]] = gml_st.tile [0, %[[ARG4]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_6]]] : tensor<1x4096xf32>[!gml_st.tile<1x1>] to f32
  //     CHECK:         %[[TILE_7:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_5:.*]] = gml_st.materialize %[[ARG5]][%[[TILE_7]]] : tensor<1x1xf32>[!gml_st.tile<1x1>] to f32
  //     CHECK:         %[[MAXF:.*]] = arith.maxf %[[MATERIALIZE_5]], %[[MATERIALIZE_4]] : f32
  //     CHECK:         %[[TILE_7_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[MAXF]] into %[[ARG5]][%[[TILE_7_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_4:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[FOR]] into %[[EMPTY_1]][%[[TILE_4]]]
  //     CHECK:     %[[REDUCE:.*]] = linalg.reduce
  //     CHECK:          ins(%[[PARALLEL_1]] : tensor<1x32xf32>)
  //     CHECK:          outs(%[[FILL]] : tensor<1xf32>)
  //     CHECK:          dimensions = [1]
  //     CHECK:     (%[[IN:.*]]: f32, %[[OUT:.*]]: f32) {
  //     CHECK:       %[[MAXF_0:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
  //     CHECK:       linalg.yield %[[MAXF_0]] : f32
  //     CHECK:     %[[MATERIALIZE_6:.*]] = gml_st.materialize %[[EMPTY_0]][%[[TILE_1]]]
  //     CHECK:     %[[TILE_8:.*]] = gml_st.tile [%[[ADDI]]] [1] [1]
  //     CHECK:     %[[MATERIALIZE_7:.*]] = gml_st.materialize %[[EMPTY]][%[[TILE_8]]]
  //     CHECK:     %[[FILL_2:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_7]] : tensor<1xf32>)
  //     CHECK:     %[[EMPTY_2:.*]] = tensor.empty()
  //     CHECK:     %[[TILE_9:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:     %[[MATERIALIZE_8:.*]] = gml_st.materialize %[[MATERIALIZE_7]][%[[TILE_9]]]
  //     CHECK:     %[[FILL_3:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_8]] : tensor<1xf32>)
  //     CHECK:     %[[EXTRACTED_1:.*]] = tensor.extract %[[FILL_3]][%[[C0]]]
  //     CHECK:     %[[PARALLEL_2:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread")
  //     CHECK:       %[[TILE_11:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       %[[MATERIALIZE_9:.*]] = gml_st.materialize %[[EMPTY_2]][%[[TILE_11]]]
  //     CHECK:       %[[FILL_4:.*]] = linalg.fill ins(%[[EXTRACTED_1]] : f32) outs(%[[MATERIALIZE_9]] : tensor<1x1xf32>)
  //     CHECK:       %[[FOR_0:.*]] = gml_st.for (%[[ARG4_0:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5_0:.*]] = %[[FILL_4]]: tensor<1x1xf32>)
  //     CHECK:         %[[TILE_12:.*]] = gml_st.tile [0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_10:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_12]]]
  //     CHECK:         %[[TILE_13:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:         %[[MATERIALIZE_11:.*]] = gml_st.materialize %[[REDUCE]][%[[TILE_13]]]
  //     CHECK:         %[[TILE_14:.*]] = gml_st.tile [0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_12:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_14]]]
  //     CHECK:         %[[BROADCAST:.*]] = linalg.broadcast
  //     CHECK:           ins(%[[MATERIALIZE_11]] : tensor<1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_12]] : tensor<1x1xf32>)
  //     CHECK:           dimensions = [1]
  //     CHECK:         %[[TILE_15:.*]] = gml_st.tile [0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_13:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_15]]]
  //     CHECK:         %[[MAP:.*]] = linalg.map
  //     CHECK:            ins(%[[MATERIALIZE_10]], %[[BROADCAST]] : tensor<1x1xf32>, tensor<1x1xf32>)
  //     CHECK:            outs(%[[MATERIALIZE_13]] : tensor<1x1xf32>)
  //     CHECK:         (%[[IN_1:.*]]: f32, %[[IN_3:.*]]: f32) {
  //     CHECK:           %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_3]] : f32
  //     CHECK:           linalg.yield %[[SUBF]] : f32
  //     CHECK:         %[[TILE_16:.*]] = gml_st.tile [0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_14:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_16]]]
  //     CHECK:         %[[MAP_0:.*]] = linalg.map
  //     CHECK:           ins(%[[MAP]] : tensor<1x1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_14]] : tensor<1x1xf32>)
  //     CHECK:         %[[IN_2:.*]]: f32) {
  //     CHECK:           %[[EXP:.*]] = math.exp %[[IN_2]] : f32
  //     CHECK:           linalg.yield %[[EXP]] : f32
  //     CHECK:         %[[EXTRACTED_2:.*]] = tensor.extract %[[MAP_0]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[TILE_17:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_15:.*]] = gml_st.materialize %[[ARG5_0]][%[[TILE_17]]] : tensor<1x1xf32>[!gml_st.tile<1x1>] to f32
  //     CHECK:         %[[ADDF:.*]] = arith.addf %[[MATERIALIZE_15]], %[[EXTRACTED_2]] : f32
  //     CHECK:         %[[TILE_17_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[ADDF]] into %[[ARG5_0]][%[[TILE_17_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_10:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[FOR_0]] into %[[EMPTY_2]][%[[TILE_10]]]
  //     CHECK:     %[[REDUCE_0:.*]] = linalg.reduce
  //     CHECK:         ins(%[[PARALLEL_2]] : tensor<1x32xf32>)
  //     CHECK:         outs(%[[FILL_2]] : tensor<1xf32>)
  //     CHECK:         dimensions = [1]
  //     CHECK:         (%[[IN_4:.*]]: f32, %[[OUT_3:.*]]: f32) {
  //     CHECK:           %[[ADDF_0:.*]] = arith.addf %[[OUT_3]], %[[IN_4]] : f32
  //     CHECK:           linalg.yield %[[ADDF_0]] : f32
  //     CHECK:     %[[PARALLEL_3:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) distribution ("thread")
  //     CHECK:       %[[SUBI:.*]] = arith.subi %[[C4127]], %[[ARG3]] : index
  //     CHECK:       %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]] : index
  //     CHECK:       %[[TILE_18:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 32]
  //     CHECK:       %[[MATERIALIZE_16:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_18]]]
  //     CHECK:       %[[FOR_1:.*]] = gml_st.for (%[[ARG4_1:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]]) outs (%[[ARG5_1:.*]] = %[[MATERIALIZE_16]]: tensor<1x?xf32>)
  //     CHECK:         %[[MULI:.*]] = arith.muli %[[ARG4_1]], %[[C32]] : index
  //     CHECK:         %[[ADDI_0:.*]] = arith.addi %[[ARG3]], %[[MULI]] : index
  //     CHECK:         %[[TILE_19:.*]] = gml_st.tile [0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_17:.*]] = gml_st.materialize %[[MATERIALIZE_0]][%[[TILE_19]]]
  //     CHECK:         %[[TILE_20:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:         %[[MATERIALIZE_18:.*]] = gml_st.materialize %[[REDUCE]][%[[TILE_20]]]
  //     CHECK:         %[[TILE_21:.*]] = gml_st.tile [0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_19:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_21]]]
  //     CHECK:         %[[BROADCAST_0:.*]] = linalg.broadcast
  //     CHECK:           ins(%[[MATERIALIZE_18]] : tensor<1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_19]] : tensor<1x1xf32>)
  //     CHECK:           dimensions = [1]
  //     CHECK:         %[[TILE_22:.*]] = gml_st.tile [0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_20:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_22]]]
  //     CHECK:         %[[MAP_1:.*]] = linalg.map
  //     CHECK:           ins(%[[MATERIALIZE_17]], %[[BROADCAST_0]] : tensor<1x1xf32>, tensor<1x1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_20]] : tensor<1x1xf32>)
  //     CHECK:         (%[[IN_6:.*]]: f32, %[[IN_4_0:.*]]: f32) {
  //     CHECK:           %[[SUBF_0:.*]] = arith.subf %[[IN_6]], %[[IN_4_0]] : f32
  //     CHECK:           linalg.yield %[[SUBF_0]] : f32
  //     CHECK:         %[[TILE_23:.*]] = gml_st.tile [0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_21:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_23]]]
  //     CHECK:         %[[MAP_2:.*]] = linalg.map
  //     CHECK:            ins(%[[MAP_1]] : tensor<1x1xf32>)
  //     CHECK:            outs(%[[MATERIALIZE_21]] : tensor<1x1xf32>)
  //     CHECK:         (%[[IN_7:.*]]: f32) {
  //     CHECK:           %[[EXP_0:.*]] = math.exp %[[IN_7]] : f32
  //     CHECK:           linalg.yield %[[EXP_0]] : f32
  //     CHECK:         %[[EXTRACTED_2_0:.*]] = tensor.extract %[[MAP_2]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[TILE_24:.*]] = gml_st.tile [0] [1] [1]
  //     CHECK:         %[[MATERIALIZE_22:.*]] = gml_st.materialize %[[REDUCE_0]][%[[TILE_24]]]
  //     CHECK:         %[[TILE_25:.*]] = gml_st.tile [0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_23:.*]] = gml_st.materialize %[[MATERIALIZE_6]][%[[TILE_25]]]
  //     CHECK:         %[[BROADCAST_1:.*]] = linalg.broadcast
  //     CHECK:             ins(%[[MATERIALIZE_22]] : tensor<1xf32>)
  //     CHECK:             outs(%[[MATERIALIZE_23]] : tensor<1x1xf32>)
  //     CHECK:             dimensions = [1]
  //     CHECK:         %[[EXTRACTED_3:.*]] = tensor.extract %[[BROADCAST_1]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[DIVF:.*]] = arith.divf %[[EXTRACTED_2_0]], %[[EXTRACTED_3]] : f32
  //     CHECK:         %[[TILE_26:.*]] = gml_st.tile [0, %[[ARG4_1]]] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[DIVF]] into %[[ARG5_1]][%[[TILE_26]]] : f32 into tensor<1x?xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_18_:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 32]
  //     CHECK:       gml_st.set_yield %[[FOR_1]] into %[[MATERIALIZE_6]][%[[TILE_18_]]]
  //     CHECK:     gml_st.set_yield %[[PARALLEL_3]] into %[[MATERIALIZE]][%[[TILE_0]]]
  //     CHECK:   gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY_0]][%[[TILE]]]
  //     CHECK: return %[[PARALLEL]]
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
  %4 = gml_st.parallel (%arg1) = (%c0) to (%c2048) step (%c1024)
      distribution ("block") {
    %5 = gml_st.tile [%arg1, 0] [1024, 4096] [1, 1] : !gml_st.tile<1024x4096>
    %6 = gml_st.materialize %2[%5]
        : tensor<2048x4096xf32>[!gml_st.tile<1024x4096>] to tensor<1024x4096xf32>
    %7 = gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c1)
        distribution ("warp") {
      %8 = gml_st.tile [%arg2, 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
      %9 = arith.addi %arg1, %arg2 : index
      %10 = gml_st.tile [%9, 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
      %11 = gml_st.materialize %arg0[%10]
          : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
      %12 = gml_st.tile [%9] [1] [1] : !gml_st.tile<1>
      %13 = gml_st.materialize %1[%12]
          : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
      %14 = linalg.reduce ins(%11 : tensor<1x4096xf32>)
                     outs(%13 : tensor<1xf32>) dimensions = [1]
        (%in: f32, %out: f32) {
            %23 = arith.maxf %out, %in : f32
            linalg.yield %23 : f32
        }
      %15 = gml_st.materialize %2[%10]
          : tensor<2048x4096xf32>[!gml_st.tile<1x4096>] to tensor<1x4096xf32>
      %16 = linalg.broadcast
          ins(%14 : tensor<1xf32>) outs(%15 : tensor<1x4096xf32>)
          dimensions = [1]
      %17 = linalg.map ins(%11, %16 : tensor<1x4096xf32>, tensor<1x4096xf32>)
          outs(%15 : tensor<1x4096xf32>)
      (%in: f32, %in_1: f32) {
        %23 = arith.subf %in, %in_1 : f32
        linalg.yield %23 : f32
      }
      %18 = linalg.map ins(%17 : tensor<1x4096xf32>)
          outs(%15 : tensor<1x4096xf32>)
      (%in: f32) {
        %23 = math.exp %in : f32
        linalg.yield %23 : f32
      }
      %19 = gml_st.materialize %3[%12]
          : tensor<2048xf32>[!gml_st.tile<1>] to tensor<1xf32>
      %20 = linalg.reduce ins(%18 : tensor<1x4096xf32>)
                outs(%19 : tensor<1xf32>) dimensions = [1]
        (%in: f32, %out: f32) {
            %23 = arith.addf %out, %in : f32
            linalg.yield %23 : f32
      }
      %21 = linalg.broadcast
          ins(%20 : tensor<1xf32>) outs(%15 : tensor<1x4096xf32>)
          dimensions = [1]
      %22 = linalg.map ins(%18, %21 : tensor<1x4096xf32>, tensor<1x4096xf32>)
          outs(%15 : tensor<1x4096xf32>)
      (%in: f32, %in_1: f32) {
        %23 = arith.divf %in, %in_1 : f32
        linalg.yield %23 : f32
      }
      gml_st.set_yield %22 into %6[%8]
          : tensor<1x4096xf32> into tensor<1024x4096xf32>[!gml_st.tile<1x4096>]
    } : tensor<1024x4096xf32>
    gml_st.set_yield %7 into %2[%5]
        : tensor<1024x4096xf32> into tensor<2048x4096xf32>[!gml_st.tile<1024x4096>]
  } : tensor<2048x4096xf32>
  return %4 : tensor<2048x4096xf32>
}
