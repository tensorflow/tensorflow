// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:     --gml-st-pipeline="tile-sizes=64,4 fuse" | \
// RUN: FileCheck %s --check-prefix=TILE-CHECK

// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:     --gml-st-pipeline="tile-sizes=1,1 fuse" | \
// RUN: FileCheck %s --check-prefix=POINT-CHECK

// TODO(akuegel): Also run with the option lower-to-loops. This fails currently
// due to not having a bufferization for gml_st.dynamic_broadcast_in_dim.

func.func @log(%arg0: tensor<512x4xf32>) -> tensor<512x4xf32> {
  %0 = mhlo.log %arg0 : tensor<512x4xf32>
  return %0 : tensor<512x4xf32>
}

// TILE-CHECK-LABEL: @log
// TILE-CHECK-SAME:  %[[ARG0:.*]]: tensor<512x4xf32>
// TILE-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// TILE-CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// TILE-CHECK-DAG:   %[[C64:.*]] = arith.constant 64
// TILE-CHECK-DAG:   %[[C512:.*]] = arith.constant 512
// TILE-CHECK:       %[[INIT:.*]] = linalg.init_tensor [512, 4]
// TILE-CHECK:       %[[SPACE:.*]] = gml_st.space [512, 4]
// TILE-CHECK:       %[[RESULT:.*]] = gml_st.parallel
// TILE-CHECK-SAME:      (%[[IV:.*]], %[[IV2:.*]]) = (%[[C0]], %[[C0]])
// TILE-CHECK-SAME:      to (%[[C512]], %[[C4]]) step (%[[C64]], %[[C4]])
// TILE-CHECK:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]] [64, 4] [1, 1]
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// TILE-CHECK:         %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// TILE-CHECK:         %[[LINALG_OP:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<64x4xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
// TILE-CHECK:           %[[LOG:.*]] = math.log %{{.*}}
// TILE-CHECK:           linalg.yield %[[LOG]]
// TILE-CHECK:         gml_st.set_yield %[[LINALG_OP]] into %[[INIT]][%[[TILE]]]
// TILE-CHECK:       return %[[RESULT]] : tensor<512x4xf32>

// POINT-CHECK-LABEL: @log
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<512x4xf32>
// POINT-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// POINT-CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// POINT-CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// POINT-CHECK-DAG:   %[[C512:.*]] = arith.constant 512
// POINT-CHECK-DAG:   %[[SPACE:.*]] = gml_st.space [512, 4]
// POINT-CHECK-DAG:   %[[INIT:.*]] = linalg.init_tensor [512, 4]
// POINT-CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
// POINT-CHECK-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// POINT-CHECK-SAME:      to (%[[C512]], %[[C4]]) step (%[[C1]], %[[C1]])
// POINT-CHECK-DAG:     %[[POINT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J]]]
// POINT-CHECK-DAG:     %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]]
// POINT-CHECK-DAG:     %[[LOG_SUB:.*]] = math.log %[[ARG_SUB]]
// POINT-CHECK:         gml_st.set_yield %[[LOG_SUB]] into %[[INIT]][%[[POINT]]]
// POINT-CHECK:       return %[[PARALLEL]]

// -----

func.func @transposed_log(%arg0: tensor<20x64xf32>) -> tensor<64x20xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} :
      (tensor<20x64xf32>) -> tensor<64x20xf32>
  %1 = mhlo.log %0 : tensor<64x20xf32>
  return %1 : tensor<64x20xf32>
}

// TILE-CHECK-LABEL: @transposed_log
// TILE-CHECK-SAME:  %[[ARG0:.*]]: tensor<20x64xf32>
// TILE-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// TILE-CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// TILE-CHECK-DAG:   %[[C20:.*]] = arith.constant 20
// TILE-CHECK-DAG:   %[[C64:.*]] = arith.constant 64
// TILE-CHECK:       %[[INIT:.*]] = linalg.init_tensor [64, 20]
// TILE-CHECK:       %[[SPACE:.*]] = gml_st.space [64, 20]
// TILE-CHECK:       %[[RESULT:.*]] = gml_st.parallel
// TILE-CHECK-SAME:      (%[[IV:.*]], %[[IV2:.*]]) = (%[[C0]], %[[C0]])
// TILE-CHECK-SAME:      to (%[[C64]], %[[C20]]) step (%[[C64]], %[[C4]])
// TILE-CHECK:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]] [64, 4] [1, 1]
// TILE-CHECK:         %[[SPACE2:.*]] = gml_st.space [20, 64] : !gml_st.tile<20x64>
// TILE-CHECK:         %[[TILE2:.*]] = gml_st.tile %[[SPACE2]] [%[[IV2]], %[[IV]]] [4, 64] [1, 1]
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE2]]]
// TILE-CHECK:         %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// TILE-CHECK:         %[[LINALG_OP:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<4x64xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
// TILE-CHECK:         %[[LOG_RES:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[LINALG_OP]] : tensor<64x4xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
// TILE-CHECK:           %[[LOG:.*]] = math.log %{{.*}}
// TILE-CHECK:           linalg.yield %[[LOG]]
// TILE-CHECK:         gml_st.set_yield %[[LOG_RES]] into %[[INIT]][%[[TILE]]]
// TILE-CHECK:       return %[[RESULT]]

// POINT-CHECK:      @transposed_log
// POINT-CHECK-SAME: %[[ARG:.*]]: tensor<20x64xf32>)
// POINT-CHECK-DAG:  %[[C0:.*]] = arith.constant 0
// POINT-CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// POINT-CHECK-DAG:  %[[C20:.*]] = arith.constant 20
// POINT-CHECK-DAG:  %[[C64:.*]] = arith.constant 64
// POINT-CHECK-DAG:  %[[SPACE:.*]] = gml_st.space [64, 20]
// POINT-CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [64, 20]
// POINT-CHECK:      %[[PARALLEL:.*]] = gml_st.parallel
// POINT-CHECK-SAME:     (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[C0]], %[[C0]])
// POINT-CHECK-SAME:     to (%[[C64]], %[[C20]]) step (%[[C1]], %[[C1]])
// POINT-CHECK-DAG:    %[[POINT:.*]] = gml_st.point %[[SPACE]] [%[[ARG1]], %[[ARG2]]]
// POINT-CHECK-DAG:    %[[TRANSPOSED_POINT:.*]] = gml_st.transpose_dims %[[POINT]], [1, 0]
// POINT-CHECK-DAG:    %[[SUB_ARG:.*]] = gml_st.materialize %[[ARG]][%[[TRANSPOSED_POINT]]]
// POINT-CHECK-DAG:    %[[SUB_LOG:.*]] = math.log %[[SUB_ARG]]
// POINT-CHECK:        gml_st.set_yield %[[SUB_LOG]] into %[[INIT]][%[[POINT]]]
// POINT-CHECK:      return %[[PARALLEL]]

// -----

func.func @broadcast_in_dim(%arg0: tensor<?xf32>, %shape: tensor<2xindex>)
    -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape)
      {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// TILE-CHECK-LABEL: @broadcast_in_dim
// TILE-CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// TILE-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// TILE-CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// TILE-CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// TILE-CHECK-DAG:   %[[C64:.*]] = arith.constant 64
// TILE-CHECK-DAG:   %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// TILE-CHECK-DAG:   %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// TILE-CHECK-DAG:   %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
// TILE-CHECK-DAG:   %[[SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// TILE-CHECK:       %[[RES:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[D0]], %[[D1]]) step (%[[C64]], %[[C4]])
// TILE-CHECK-DAG:     %[[I_PLUS_64:.*]] = arith.addi %[[I]], %[[C64]]
// TILE-CHECK-DAG:     %[[IS_PARTIAL0:.*]] = arith.cmpi sgt, %[[I_PLUS_64]], %[[D0]]
// TILE-CHECK-DAG:     %[[D0_MINUS_I:.*]] = arith.subi %[[D0]], %[[I]]
// TILE-CHECK-DAG:     %[[TILE_SIZE0:.*]] = arith.select %[[IS_PARTIAL0]], %[[D0_MINUS_I]], %[[C64]]
// TILE-CHECK-DAG:     %[[J_PLUS_4:.*]] = arith.addi %[[J]], %[[C4]]
// TILE-CHECK-DAG:     %[[IS_PARTIAL1:.*]] = arith.cmpi sgt, %[[J_PLUS_4]], %[[D1]]
// TILE-CHECK-DAG:     %[[D1_MINUS_J:.*]] = arith.subi %[[D1]], %[[J]]
// TILE-CHECK-DAG:     %[[TILE_SIZE1:.*]] = arith.select %[[IS_PARTIAL1]], %[[D1_MINUS_J]], %[[C4]]
// TILE-CHECK-DAG:     %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[TILE_SIZE0]], %[[TILE_SIZE1]]] [1, 1]
// TILE-CHECK-DAG:     %[[ARG_D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// TILE-CHECK-DAG:     %[[ARG_SPACE:.*]] = gml_st.space [%[[ARG_D0]]]
// TILE-CHECK-DAG:     %[[IS_EXPANDING0:.*]] = arith.cmpi ne, %[[ARG_D0]], %[[D1]]
// TILE-CHECK-DAG:     %[[ARG_TILE_OFFSET0:.*]] = arith.select %[[IS_EXPANDING0]], %[[C0]], %[[J]]
// TILE-CHECK-DAG:     %[[ARG_TILE_SIZE0:.*]] = arith.select %[[IS_EXPANDING0]], %[[C1]], %[[TILE_SIZE1]]
// TILE-CHECK-DAG:     %[[ARG_TILE:.*]] = gml_st.tile %[[ARG_SPACE]] [%[[ARG_TILE_OFFSET0]]] [%[[ARG_TILE_SIZE0]]] [1]
// TILE-CHECK-DAG:     %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// TILE-CHECK-DAG:     %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE]]]
// TILE-CHECK-DAG:     %[[BCAST_SUB:.*]] = gml_st.dynamic_broadcast_in_dim ins(%[[ARG_SUB]] : tensor<?xf32>) outs(%[[INIT_SUB]] : tensor<?x?xf32>) {broadcast_dimensions = [:i64 1]}
// TILE-CHECK:         gml_st.set_yield %[[BCAST_SUB]] into %[[INIT]][%[[TILE]]]
// TILE-CHECK:       return %[[RES]]

// POINT-CHECK-LABEL: @broadcast_in_dim
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// POINT-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// POINT-CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// POINT-CHECK-DAG:   %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// POINT-CHECK-DAG:   %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// POINT-CHECK-DAG:   %[[RES_SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// POINT-CHECK-DAG:   %[[RES_INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
// POINT-CHECK:       %[[PARALLEL:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[D0]], %[[D1]]) step (%[[C1]], %[[C1]])
// POINT-CHECK-DAG:     %[[RES_POINT:.*]] = gml_st.point %[[RES_SPACE]] [%[[I]], %[[J]]]
// POINT-CHECK-DAG:     %[[ARG_D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// POINT-CHECK-DAG:     %[[ARG_SPACE:.*]] = gml_st.space [%[[ARG_D0]]]
// POINT-CHECK-DAG:     %[[DD_RES_POINT:.*]] = gml_st.drop_dims %[[RES_POINT]], [1]
// POINT-CHECK-DAG:     %[[EXPANDING_D0:.*]] = arith.cmpi ne, %[[ARG_D0]], %[[D1]]
// POINT-CHECK-DAG:     %[[DD_RES_POINT_OFFSET_D0_D0:.*]] = gml_st.offset %[[DD_RES_POINT]][%[[C0]]]
// POINT-CHECK-DAG:     %[[OFFSET_D0:.*]] = arith.select %[[EXPANDING_D0]], %[[C0]], %[[DD_RES_POINT_OFFSET_D0_D0]]
// POINT-CHECK-DAG:     %[[ARG_RES_POINT:.*]] = gml_st.point %[[ARG_SPACE]] [%[[OFFSET_D0]]]
// POINT-CHECK-DAG:     %[[BCAST_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_RES_POINT]]]
// POINT-CHECK:         gml_st.set_yield %[[BCAST_SUB]] into %[[RES_INIT]][%[[RES_POINT]]]
// POINT-CHECK:       return %[[PARALLEL]]

// -----

func.func @log_log_bcast(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>)
    -> tensor<?x?xf32> {
  %0 = mhlo.log %arg0 : tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %arg1)
      {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
      : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = mhlo.log %1 : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// TILE-CHECK-LABEL: @log_log_bcast
// TILE-CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// TILE-CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// TILE-CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// TILE-CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// TILE-CHECK-DAG:   %[[C64:.*]] = arith.constant 64
// TILE-CHECK-DAG:   %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// TILE-CHECK-DAG:   %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// TILE-CHECK-DAG:   %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
// TILE-CHECK-DAG:   %[[ARG_D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// TILE-CHECK-DAG:   %[[ARG_D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// TILE-CHECK-DAG:   %[[ARG_INIT:.*]] = linalg.init_tensor [%[[ARG_D0]], %[[ARG_D1]]]
// TILE-CHECK-DAG:   %[[SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// TILE-CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
// TILE-CHECK-SAME:      (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// TILE-CHECK-SAME:      to (%[[D0]], %[[D1]])
// TILE-CHECK-SAME:      step (%[[C64]], %[[C4]])
// TILE-CHECK-DAG:     %[[I_PLUS_64:.*]] = arith.addi %[[I]], %[[C64]]
// TILE-CHECK-DAG:     %[[IS_PARTIAL0:.*]] = arith.cmpi sgt, %[[I_PLUS_64]], %[[D0]]
// TILE-CHECK-DAG:     %[[D0_MINUS_I:.*]] = arith.subi %[[D0]], %[[I]]
// TILE-CHECK-DAG:     %[[TILE_SIZE0:.*]] = arith.select %[[IS_PARTIAL0]], %[[D0_MINUS_I]], %[[C64]]
// TILE-CHECK-DAG:     %[[J_PLUS_4:.*]] = arith.addi %[[J]], %[[C4]]
// TILE-CHECK-DAG:     %[[IS_PARTIAL1:.*]] = arith.cmpi sgt, %[[J_PLUS_4]], %[[D1]]
// TILE-CHECK-DAG:     %[[D1_MINUS_J:.*]] = arith.subi %[[D1]], %[[J]]
// TILE-CHECK-DAG:     %[[TILE_SIZE1:.*]] = arith.select %[[IS_PARTIAL1]], %[[D1_MINUS_J]], %[[C4]]
// TILE-CHECK-DAG:     %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [%[[TILE_SIZE0]], %[[TILE_SIZE1]]] [1, 1]
// TILE-CHECK-DAG:     %[[ARG_SPACE:.*]] = gml_st.space [%[[ARG_D0]], %[[ARG_D1]]]
// TILE-CHECK-DAG:     %[[IS_EXPANDING0:.*]] = arith.cmpi ne, %[[ARG_D0]], %[[D0]]
// TILE-CHECK-DAG:     %[[IS_EXPANDING1:.*]] = arith.cmpi ne, %[[ARG_D1]], %[[D1]]
// TILE-CHECK-DAG:     %[[ARG_TILE_OFFSET0:.*]] = arith.select %[[IS_EXPANDING0]], %[[C0]], %[[I]]
// TILE-CHECK-DAG:     %[[ARG_TILE_OFFSET1:.*]] = arith.select %[[IS_EXPANDING1]], %[[C0]], %[[J]]
// TILE-CHECK-DAG:     %[[ARG_TILE_SIZE0:.*]] = arith.select %[[IS_EXPANDING0]], %[[C1]], %[[TILE_SIZE0]]
// TILE-CHECK-DAG:     %[[ARG_TILE_SIZE1:.*]] = arith.select %[[IS_EXPANDING1]], %[[C1]], %[[TILE_SIZE1]]
// TILE-CHECK-DAG:     %[[ARG_TILE:.*]] = gml_st.tile %[[ARG_SPACE]] [%[[ARG_TILE_OFFSET0]], %[[ARG_TILE_OFFSET1]]] [%[[ARG_TILE_SIZE0]], %[[ARG_TILE_SIZE1]]] [1, 1]
// TILE-CHECK-DAG:     %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// TILE-CHECK-DAG:     %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE]]]
// TILE-CHECK-DAG:     %[[ARG_INIT_SUB:.*]] = gml_st.materialize %[[ARG_INIT]][%[[ARG_TILE]]]
// TILE-CHECK:         %[[GENERIC_SUB0:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        outs(%[[ARG_INIT_SUB]] : tensor<?x?xf32>)
// TILE-CHECK:         %[[BCAST_SUB:.*]] = gml_st.dynamic_broadcast_in_dim
// TILE-CHECK-SAME:        ins(%[[GENERIC_SUB0]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        {broadcast_dimensions = [:i64 0, 1]}
// TILE-CHECK:         %[[GENERIC_SUB1:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[BCAST_SUB]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// TILE-CHECK:         gml_st.set_yield %[[GENERIC_SUB1]] into %[[INIT]][%[[TILE]]]
// TILE-CHECK:       return %[[PARALLEL]]


// POINT-CHECK-LABEL: @log_log_bcast
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// POINT-CHECK:       %[[C0:.*]] = arith.constant 0
// POINT-CHECK:       %[[C1:.*]] = arith.constant 1
// POINT-CHECK:       %[[D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// POINT-CHECK:       %[[D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// POINT-CHECK:       %[[S0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// POINT-CHECK:       %[[S1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// POINT-CHECK:       %[[RES_SPACE:.*]] = gml_st.space [%[[S0]], %[[S1]]]
// POINT-CHECK:       %[[INIT:.*]] = linalg.init_tensor [%[[S0]], %[[S1]]]
// POINT-CHECK:       %[[RES:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[S0]], %[[S1]]) step (%[[C1]], %[[C1]])
// POINT-CHECK:         %[[POINT:.*]] = gml_st.point %[[RES_SPACE]] [%[[I]], %[[J]]]
// POINT-CHECK:         %[[SPACE_ARG:.*]] = gml_st.space [%[[D0]], %[[D1]]]
// POINT-CHECK:         %[[POINT_ARG:.*]] = gml_st.point %[[SPACE_ARG]] [%{{.*}}, %{{.*}}]
// POINT-CHECK:         %[[SUB_ARG:.*]] = gml_st.materialize %[[ARG]][%[[POINT_ARG]]]
// POINT-CHECK:         %[[LOG:.*]] = math.log %[[SUB_ARG]]
// POINT-CHECK:         %[[LOG_LOG:.*]] = math.log %[[LOG]]
// POINT-CHECK:         gml_st.set_yield %[[LOG_LOG]] into %[[INIT]][%[[POINT]]]
// POINT-CHECK:       return %[[RES]]
