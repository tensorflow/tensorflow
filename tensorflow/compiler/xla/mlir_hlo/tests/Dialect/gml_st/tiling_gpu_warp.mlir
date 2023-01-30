// RUN: mlir-hlo-opt %s --split-input-file --gml-tiling-gpu-warp | \
// RUN: FileCheck %s

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
  //     CHECK: %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C1024]]) step (%[[C1]])
 // CHECK-SAME:     outs (%[[EMPTY_:.*]] = %[[EMPTY]]:
 // CHECK-SAME:     distribution ("warp")
  //     CHECK:   %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0] [1, 13] [1, 1]
  //     CHECK:   %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[EMPTY_]][%[[ARG1]]] [1] [1]
  //     CHECK:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_0]] : tensor<1xf32>)
  //     CHECK:   %[[EMPTY_0:.*]] = tensor.empty()
  //     CHECK:   %[[EXTRACTED:.*]] = tensor.extract %[[FILL]][%[[C0]]]
  //     CHECK:   %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C16]]) step (%[[C1]])
 // CHECK-SAME:       outs (%[[EMPTY_0_:.*]] = %[[EMPTY_0]]
 // CHECK-SAME:       distribution ("thread")
  //     CHECK:     %[[MATERIALIZE_2:.*]] = tensor.extract_slice %[[EMPTY_0_]][0, %[[ARG2]]] [1, 1] [1, 1]
  //     CHECK:     %[[FILL_1:.*]] = linalg.fill ins(%[[EXTRACTED]] : f32) outs(%[[MATERIALIZE_2]] : tensor<1x1xf32>)
  //     CHECK:     %[[FOR:.*]] = gml_st.for (%[[ARG3:.*]]) = (%[[ARG2]]) to (%[[C13]]) step (%[[C16]]) outs (%[[ARG4:.*]] = %[[FILL_1]]: tensor<1x1xf32>)
  //     CHECK:       %[[MATERIALIZE_3_:.*]] = tensor.extract_slice %[[MATERIALIZE]][0, %[[ARG3]]] [1, 1] [1, 1] : tensor<1x13xf32> to tensor<1x1xf32>
  //     CHECK:       %[[MATERIALIZE_3:.*]] = tensor.extract %[[MATERIALIZE_3_]]
  //     CHECK:       %[[MATERIALIZE_4:.*]] = tensor.extract %[[ARG4]]
  //     CHECK:       %[[MAXF:.*]] = arith.maxf %[[MATERIALIZE_4]], %[[MATERIALIZE_3]] : f32
  //     CHECK:       %[[TILE_6_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[MAXF]] into %[[ARG4]][%[[TILE_6_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:     %[[TILE_3:.*]] = gml_st.tile [0, %[[ARG2]]] [1, 1] [1, 1]
  //     CHECK:     gml_st.set_yield %[[FOR]] into %[[EMPTY_0_]][%[[TILE_3]]]
  //     CHECK:   %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  //     CHECK:       ins(%[[PARALLEL_0]] : tensor<1x16xf32>)
  //     CHECK:       outs(%[[FILL]] : tensor<1xf32>)
  //     CHECK:       dimensions = [1]
  //     CHECK:   %[[TILE_0:.*]] = gml_st.tile [%[[ARG1]]] [1] [1]
  //     CHECK:   gml_st.set_yield %[[REDUCE]] into %[[EMPTY_]][%[[TILE_0]]]
  //     CHECK: return %[[PARALLEL]]
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<7xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7xf32>)
      -> tensor<7xf32>
  %2 = gml_st.parallel (%arg1) = (%c0) to (%c1024) step (%c1)
      outs (%out_ = %1: tensor<7xf32>)
      distribution ("warp") {
    %4 = tensor.extract_slice %arg0[%arg1, 0] [1, 13] [1, 1]
        : tensor<7x13xf32> to tensor<1x13xf32>
    %6 = tensor.extract_slice %out_[%arg1] [1] [1]
        : tensor<7xf32> to tensor<1xf32>
    %7 = linalg.reduce { arith.maxf }
           ins(%4 : tensor<1x13xf32>)
           outs(%6 : tensor<1xf32>)
           dimensions = [1]
    %5 = gml_st.tile [%arg1] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %7 into %out_[%5]
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
  // CHECK-SAME:     step (%[[C1]]) outs (%[[EMPTY_:.*]] = %[[EMPTY]]:
  // CHECK-SAME:     distribution ("warp")
  // CHECK:        %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0] [1, 13] [1, 1]
  // CHECK:        %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], 0] [1, 13] [1, 1]
  // CHECK:        %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[EMPTY_]][%[[ARG2]], 0] [1, 13] [1, 1]
  // CHECK:        %[[PARALLEL_0:.*]] = gml_st.parallel
  // CHECK-SAME:       (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C16]])
  // CHECK-SAME:       step (%[[C1]]) outs (%[[MATERIALIZE_1_:.*]] = %[[MATERIALIZE_1]]:
  // CHECK-SAME:       distribution ("thread")
  // CHECK:          %[[SUBI:.*]] = arith.subi %[[C28]], %[[ARG3]]
  // CHECK:          %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C16]]
  // CHECK:          %[[MATERIALIZE_2:.*]] = tensor.extract_slice %[[MATERIALIZE_1_]][0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 16]
  // CHECK:          %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]]) = (%[[C0]])
  // CHECK-SAME:         to (%[[DIVUI]]) step (%[[C1]])
  // CHECK-SAME:         outs (%[[ARG5:.*]] = %[[MATERIALIZE_2]]: tensor<1x?xf32>)
  // CHECK:            %[[MULI:.*]] = arith.muli %[[ARG4]], %[[C16]] : index
  // CHECK:            %[[ADDI:.*]] = arith.addi %[[ARG3]], %[[MULI]] : index
  // CHECK:            %[[MATERIALIZE_3_:.*]] = tensor.extract_slice %[[MATERIALIZE]][0, %[[ADDI]]] [1, 1] [1, 1]
  // CHECK:            %[[MATERIALIZE_3:.*]] = tensor.extract %[[MATERIALIZE_3_]]
  // CHECK:            %[[MATERIALIZE_4_:.*]] = tensor.extract_slice %[[MATERIALIZE_0]][0, %[[ADDI]]] [1, 1] [1, 1]
  // CHECK:            %[[MATERIALIZE_4:.*]] = tensor.extract %[[MATERIALIZE_4_]]
  // CHECK:            %[[SUBF:.*]] = arith.subf %[[MATERIALIZE_3]], %[[MATERIALIZE_4]]
  // CHECK:            %[[TILE_2:.*]] = gml_st.tile [0, %[[ARG4]]] [1, 1] [1, 1]
  // CHECK:            gml_st.set_yield %[[SUBF]] into %[[ARG5]][%[[TILE_2]]]
  // CHECK:          %[[TILE_0:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 16]
  // CHECK:          gml_st.set_yield %[[FOR]] into %[[MATERIALIZE_1_]][%[[TILE_0]]]
  // CHECK:        %[[TILE:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 13] [1, 1]
  // CHECK:        gml_st.set_yield %[[PARALLEL_0]] into %[[EMPTY_]][%[[TILE]]]
  // CHECK:      return %[[PARALLEL]]
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<7x13xf32>
  %1 = gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c1)
      outs (%out_ = %0: tensor<7x13xf32>) distribution ("warp") {
    %3 = tensor.extract_slice %arg0 [%arg2, 0] [1, 13] [1, 1]
        : tensor<7x13xf32> to tensor<1x13xf32>
    %4 = tensor.extract_slice %arg1 [%arg2, 0] [1, 13] [1, 1]
        : tensor<7x13xf32> to tensor<1x13xf32>
    %5 = tensor.extract_slice %out_ [%arg2, 0] [1, 13] [1, 1]
        : tensor<7x13xf32> to tensor<1x13xf32>
    %6 = linalg.map { arith.subf }
           ins(%3, %4 : tensor<1x13xf32>, tensor<1x13xf32>)
           outs(%5 : tensor<1x13xf32>)
    %2 = gml_st.tile [%arg2, 0] [1, 13] [1, 1] : !gml_st.tile<1x13>
    gml_st.set_yield %6 into %out_[%2]
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
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<2048xf32>
  // CHECK-DAG: %[[EMPTY_0:.*]] = tensor.empty() : tensor<2048x4096xf32>
  //     CHECK: %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C2048]]) step (%[[C1024]])
  //     CHECK-SAME:  outs (%[[BLOCK_OUT_:.*]] = %[[EMPTY_0]]:
  //     CHECK-SAME:  distribution ("block")
  //     CHECK:   %[[MATERIALIZE:.*]] = tensor.extract_slice %[[BLOCK_OUT_]][%[[ARG1]], 0] [1024, 4096] [1, 1]

  //     CHECK:   %[[PARALLEL_0:.*]] = gml_st.parallel (%[[ARG2:.*]]) = (%[[C0]]) to (%[[C1024]]) step (%[[C1]])
  //     CHECK-SAME:  outs (%[[WARP_OUT_:.*]] = %[[MATERIALIZE]]:
  //     CHECK-SAME:  distribution ("warp")
  //     CHECK:     %[[ADDI:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
  //     CHECK:     %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG0]][%[[ADDI]], 0] [1, 4096] [1, 1]
  //     CHECK:     %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[EMPTY]][%[[ADDI]]] [1] [1]
  //     CHECK:     %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[MATERIALIZE_1]] : tensor<1xf32>)
  //     CHECK:     %[[EMPTY_1:.*]] = tensor.empty()
  //     CHECK:     %[[EXTRACTED:.*]] = tensor.extract %[[FILL]][%[[C0]]]
  //     CHECK:     %[[PARALLEL_1:.*]] = gml_st.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]])
  //     CHECK-SAME:    outs (%[[THREAD_OUT0_:.*]] = %[[EMPTY_1]]
  //     CHECK-SAME:    distribution ("thread")
  //     CHECK:       %[[MATERIALIZE_3:.*]] = tensor.extract_slice %[[THREAD_OUT0_]][0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       %[[FILL_1:.*]] = linalg.fill ins(%[[EXTRACTED]] : f32) outs(%[[MATERIALIZE_3]] : tensor<1x1xf32>)
  //     CHECK:       %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5:.*]] = %[[FILL_1]]: tensor<1x1xf32>)
  //     CHECK:         %[[MATERIALIZE_4_:.*]] = tensor.extract_slice %[[MATERIALIZE_0]][0, %[[ARG4]]] [1, 1] [1, 1] : tensor<1x4096xf32> to tensor<1x1xf32>
  //     CHECK:         %[[MATERIALIZE_4:.*]] = tensor.extract %[[MATERIALIZE_4_]]
  //     CHECK:         %[[MATERIALIZE_5:.*]] = tensor.extract %[[ARG5]]
  //     CHECK:         %[[MAXF:.*]] = arith.maxf %[[MATERIALIZE_5]], %[[MATERIALIZE_4]] : f32
  //     CHECK:         %[[TILE_7_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[MAXF]] into %[[ARG5]][%[[TILE_7_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_4:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[FOR]] into %[[THREAD_OUT0_]][%[[TILE_4]]]
  //     CHECK:     %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  //     CHECK:          ins(%[[PARALLEL_1]] : tensor<1x32xf32>)
  //     CHECK:          outs(%[[FILL]] : tensor<1xf32>)
  //     CHECK:          dimensions = [1]
  //     CHECK:     %[[MATERIALIZE_6:.*]] = tensor.extract_slice %[[EMPTY_0]][%[[ADDI]], 0] [1, 4096] [1, 1]
  //     CHECK:     %[[MATERIALIZE_7:.*]] = tensor.extract_slice %[[EMPTY]][%[[ADDI]]] [1] [1]
  //     CHECK:     %[[FILL_2:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[MATERIALIZE_7]] : tensor<1xf32>)
  //     CHECK:     %[[EMPTY_2:.*]] = tensor.empty()
  //     CHECK:     %[[EXTRACTED_1:.*]] = tensor.extract %[[FILL_2]][%[[C0]]]
  //     CHECK:     %[[PARALLEL_2:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]])
  //     CHECK-SAME:    outs (%[[THREAD_OUT1_:.*]] = %[[EMPTY_2]]
  //     CHECK-SAME:    distribution ("thread")
  //     CHECK:       %[[MATERIALIZE_9:.*]] = tensor.extract_slice %[[THREAD_OUT1_]][0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       %[[FILL_4:.*]] = linalg.fill ins(%[[EXTRACTED_1]] : f32) outs(%[[MATERIALIZE_9]] : tensor<1x1xf32>)
  //     CHECK:       %[[FOR_0:.*]] = gml_st.for (%[[ARG4_0:.*]]) = (%[[ARG3]]) to (%[[C4096]]) step (%[[C32]]) outs (%[[ARG5_0:.*]] = %[[FILL_4]]: tensor<1x1xf32>)
  //     CHECK:         %[[MATERIALIZE_10:.*]] = tensor.extract_slice %[[MATERIALIZE_0]][0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_12:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[BROADCAST:.*]] = linalg.broadcast
  //     CHECK:           ins(%[[REDUCE]] : tensor<1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_12]] : tensor<1x1xf32>)
  //     CHECK:           dimensions = [1]
  //     CHECK:         %[[MATERIALIZE_13:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MAP:.*]] = linalg.map { arith.subf }
  //     CHECK:            ins(%[[MATERIALIZE_10]], %[[BROADCAST]] : tensor<1x1xf32>, tensor<1x1xf32>)
  //     CHECK:            outs(%[[MATERIALIZE_13]] : tensor<1x1xf32>)
  //     CHECK:         %[[MATERIALIZE_14:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ARG4_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MAP_0:.*]] = linalg.map { math.exp }
  //     CHECK:           ins(%[[MAP]] : tensor<1x1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_14]] : tensor<1x1xf32>)
  //     CHECK:         %[[EXTRACTED_2:.*]] = tensor.extract %[[MAP_0]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[MATERIALIZE_15:.*]] = tensor.extract %[[ARG5_0]]

  //     CHECK:         %[[ADDF:.*]] = arith.addf %[[MATERIALIZE_15]], %[[EXTRACTED_2]] : f32
  //     CHECK:         %[[TILE_17_:.*]] = gml_st.tile [0, 0] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[ADDF]] into %[[ARG5_0]][%[[TILE_17_]]] : f32 into tensor<1x1xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_10:.*]] = gml_st.tile [0, %[[ARG3]]] [1, 1] [1, 1]
  //     CHECK:       gml_st.set_yield %[[FOR_0]] into %[[THREAD_OUT1_]][%[[TILE_10]]]
  //     CHECK:     %[[REDUCE_0:.*]] = linalg.reduce { arith.addf }
  //     CHECK:         ins(%[[PARALLEL_2]] : tensor<1x32xf32>)
  //     CHECK:         outs(%[[FILL_2]] : tensor<1xf32>)
  //     CHECK:         dimensions = [1]
  //     CHECK:     %[[PARALLEL_3:.*]] = gml_st.parallel (%[[ARG3]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]])
  //     CHECK-SAME:    outs (%[[THREAD_OUT2_:.*]] = %[[MATERIALIZE_6]]:
  //     CHECK-SAME:    distribution ("thread")
  //     CHECK:       %[[SUBI:.*]] = arith.subi %[[C4127]], %[[ARG3]] : index
  //     CHECK:       %[[DIVUI:.*]] = arith.divui %[[SUBI]], %[[C32]] : index
  //     CHECK:       %[[MATERIALIZE_16:.*]] = tensor.extract_slice %[[THREAD_OUT2_]][0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 32]
  //     CHECK:       %[[FOR_1:.*]] = gml_st.for (%[[ARG4_1:.*]]) = (%[[C0]]) to (%[[DIVUI]]) step (%[[C1]]) outs (%[[ARG5_1:.*]] = %[[MATERIALIZE_16]]: tensor<1x?xf32>)
  //     CHECK:         %[[MULI:.*]] = arith.muli %[[ARG4_1]], %[[C32]] : index
  //     CHECK:         %[[ADDI_0:.*]] = arith.addi %[[ARG3]], %[[MULI]] : index
  //     CHECK:         %[[MATERIALIZE_17:.*]] = tensor.extract_slice %[[MATERIALIZE_0]][0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MATERIALIZE_19:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[BROADCAST_0:.*]] = linalg.broadcast
  //     CHECK:           ins(%[[REDUCE]] : tensor<1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_19]] : tensor<1x1xf32>)
  //     CHECK:           dimensions = [1]
  //     CHECK:         %[[MATERIALIZE_20:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MAP_1:.*]] = linalg.map { arith.subf }
  //     CHECK:           ins(%[[MATERIALIZE_17]], %[[BROADCAST_0]] : tensor<1x1xf32>, tensor<1x1xf32>)
  //     CHECK:           outs(%[[MATERIALIZE_20]] : tensor<1x1xf32>)
  //     CHECK:         %[[MATERIALIZE_21:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[MAP_2:.*]] = linalg.map { math.exp }
  //     CHECK:            ins(%[[MAP_1]] : tensor<1x1xf32>)
  //     CHECK:            outs(%[[MATERIALIZE_21]] : tensor<1x1xf32>)
  //     CHECK:         %[[EXTRACTED_2_0:.*]] = tensor.extract %[[MAP_2]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[MATERIALIZE_23:.*]] = tensor.extract_slice %[[MATERIALIZE_6]][0, %[[ADDI_0]]] [1, 1] [1, 1]
  //     CHECK:         %[[BROADCAST_1:.*]] = linalg.broadcast
  //     CHECK:             ins(%[[REDUCE_0]] : tensor<1xf32>)
  //     CHECK:             outs(%[[MATERIALIZE_23]] : tensor<1x1xf32>)
  //     CHECK:             dimensions = [1]
  //     CHECK:         %[[EXTRACTED_3:.*]] = tensor.extract %[[BROADCAST_1]][%[[C0]], %[[C0]]]
  //     CHECK:         %[[DIVF:.*]] = arith.divf %[[EXTRACTED_2_0]], %[[EXTRACTED_3]] : f32
  //     CHECK:         %[[TILE_26:.*]] = gml_st.tile [0, %[[ARG4_1]]] [1, 1] [1, 1]
  //     CHECK:         gml_st.set_yield %[[DIVF]] into %[[ARG5_1]][%[[TILE_26]]] : f32 into tensor<1x?xf32>[!gml_st.tile<1x1>]
  //     CHECK:       %[[TILE_18_:.*]] = gml_st.tile [0, %[[ARG3]]] [1, %[[DIVUI]]] [1, 32]
  //     CHECK:       gml_st.set_yield %[[FOR_1]] into %[[THREAD_OUT2_]][%[[TILE_18_]]]
  //     CHECK:     %[[TILE_0:.*]] = gml_st.tile [%[[ARG2]], 0] [1, 4096] [1, 1]
  //     CHECK:     gml_st.set_yield %[[PARALLEL_3]] into %[[WARP_OUT_]][%[[TILE_0]]]
  //     CHECK:   %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [1024, 4096] [1, 1]
  //     CHECK:   gml_st.set_yield %[[PARALLEL_0]] into %[[BLOCK_OUT_]][%[[TILE]]]
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
      outs (%block_out_ = %2: tensor<2048x4096xf32>) distribution ("block") {
    %6 = tensor.extract_slice %block_out_[%arg1, 0] [1024, 4096] [1, 1]
        : tensor<2048x4096xf32> to tensor<1024x4096xf32>
    %7 = gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c1)
       outs (%warp_out_ = %6: tensor<1024x4096xf32>) distribution ("warp") {
      %9 = arith.addi %arg1, %arg2 : index
      %11 = tensor.extract_slice %arg0[%9, 0] [1, 4096] [1, 1]
          : tensor<2048x4096xf32> to tensor<1x4096xf32>
      %13 = tensor.extract_slice %1[%9] [1] [1]
          : tensor<2048xf32> to tensor<1xf32>
      %14 = linalg.reduce { arith.maxf }
              ins(%11 : tensor<1x4096xf32>)
              outs(%13 : tensor<1xf32>) dimensions = [1]
      %15 = tensor.extract_slice %2[%9, 0] [1, 4096] [1, 1]
          : tensor<2048x4096xf32> to tensor<1x4096xf32>
      %16 = linalg.broadcast
              ins(%14 : tensor<1xf32>) outs(%15 : tensor<1x4096xf32>)
              dimensions = [1]
      %17 = linalg.map  { arith.subf }
              ins(%11, %16 : tensor<1x4096xf32>, tensor<1x4096xf32>)
              outs(%15 : tensor<1x4096xf32>)
      %18 = linalg.map { math.exp }
              ins(%17 : tensor<1x4096xf32>)
              outs(%15 : tensor<1x4096xf32>)
      %19 = tensor.extract_slice %3[%9] [1] [1]
        : tensor<2048xf32> to tensor<1xf32>
      %20 = linalg.reduce { arith.addf }
              ins(%18 : tensor<1x4096xf32>)
              outs(%19 : tensor<1xf32>) dimensions = [1]
      %21 = linalg.broadcast
          ins(%20 : tensor<1xf32>) outs(%15 : tensor<1x4096xf32>)
          dimensions = [1]
      %22 = linalg.map { arith.divf }
              ins(%18, %21 : tensor<1x4096xf32>, tensor<1x4096xf32>)
              outs(%15 : tensor<1x4096xf32>)
      %8 = gml_st.tile [%arg2, 0] [1, 4096] [1, 1] : !gml_st.tile<1x4096>
      gml_st.set_yield %22 into %warp_out_[%8]
          : tensor<1x4096xf32> into tensor<1024x4096xf32>[!gml_st.tile<1x4096>]
    } : tensor<1024x4096xf32>
    %5 = gml_st.tile [%arg1, 0] [1024, 4096] [1, 1] : !gml_st.tile<1024x4096>
    gml_st.set_yield %7 into %block_out_[%5]
        : tensor<1024x4096xf32> into tensor<2048x4096xf32>[!gml_st.tile<1024x4096>]
  } : tensor<2048x4096xf32>
  return %4 : tensor<2048x4096xf32>
}
