// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:  --gml-st-pipeline="tile-sizes=64,4 fuse" \
// RUN: | FileCheck %s

// TODO(akuegel): Also run with the option lower-to-loops. This fails currently
// due to not having a bufferization for gml_st.dynamic_broadcast_in_dim.

// CHECK-LABEL: @log
// CHECK-SAME:  %[[ARG0:.*]]: tensor<512x4xf32>
func.func @log(%arg0: tensor<512x4xf32>) -> tensor<512x4xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4
  // CHECK-DAG:  %[[C64:.*]] = arith.constant 64
  // CHECK-DAG:  %[[C512:.*]] = arith.constant 512
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [512, 4]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [512, 4]
  // CHECK:      %[[RESULT:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[IV:.*]], %[[IV2:.*]]) = (%[[C0]], %[[C0]])
  // CHECK-SAME:     to (%[[C512]], %[[C4]]) step (%[[C64]], %[[C4]])
  // CHECK:        %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]] [64, 4] [1, 1]
  // CHECK:        %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
  // CHECK:        %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:        %[[LINALG_OP:.*]] = linalg.generic
  // CHECK-SAME:       ins(%[[ARG_SUB]] : tensor<64x4xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
  // CHECK:          %[[LOG:.*]] = math.log %{{.*}}
  // CHECK:          linalg.yield %[[LOG]]
  // CHECK:        gml_st.set_yield %[[LINALG_OP]] into %[[INIT]][%[[TILE]]]
  // CHECK:      return %[[RESULT]] : tensor<512x4xf32>
  %0 = mhlo.log %arg0 : tensor<512x4xf32>
  return %0 : tensor<512x4xf32>
}

// -----

// CHECK-LABEL: @transposed_log
// CHECK-SAME:  %[[ARG0:.*]]: tensor<20x64xf32>
func.func @transposed_log(%arg0: tensor<20x64xf32>) -> tensor<64x20xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4
  // CHECK-DAG:  %[[C20:.*]] = arith.constant 20
  // CHECK-DAG:  %[[C64:.*]] = arith.constant 64
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [64, 20]
  // CHECK:      %[[SPACE:.*]] = gml_st.space [64, 20]
  // CHECK:      %[[RESULT:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[IV:.*]], %[[IV2:.*]]) = (%[[C0]], %[[C0]])
  // CHECK-SAME:     to (%[[C64]], %[[C20]]) step (%[[C64]], %[[C4]])
  // CHECK:        %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]] [64, 4] [1, 1]
  // CHECK:        %[[SPACE2:.*]] = gml_st.space [20, 64] : !gml_st.tile<20x64>
  // CHECK:        %[[TILE2:.*]] = gml_st.tile %[[SPACE2]] [%[[IV2]], %[[IV]]] [4, 64] [1, 1]
  // CHECK:        %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE2]]]
  // CHECK:        %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:        %[[LINALG_OP:.*]] = linalg.generic
  // CHECK-SAME:       ins(%[[ARG_SUB]] : tensor<4x64xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
  // CHECK:        %[[LOG_RES:.*]] = linalg.generic
  // CHECK-SAME:       ins(%[[LINALG_OP]] : tensor<64x4xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
  // CHECK:          %[[LOG:.*]] = math.log %{{.*}}
  // CHECK:          linalg.yield %[[LOG]]
  // CHECK:        gml_st.set_yield %[[LOG_RES]] into %[[INIT]][%[[TILE]]]
  // CHECK:      return %[[RESULT]]
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} :
      (tensor<20x64xf32>) -> tensor<64x20xf32>
  %1 = mhlo.log %0 : tensor<64x20xf32>
  return %1 : tensor<64x20xf32>
}

// -----

// CHECK-LABEL: @broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>, %{{.*}}: tensor<2xindex>
func.func @broadcast_in_dim(%arg0: tensor<?xf32>, %shape: tensor<2xindex>)
    -> tensor<?x?xf32> {
  // CHECK-NOT:  dynamic_broadcast_in_dim
  // CHECK:      %[[RESULT:.*]] = gml_st.parallel
  // CHECK-DAG:    %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT:.*]][%[[BCAST_TILE:13]]]
  // CHECK-DAG:    %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE:.*]]]
  // CHECK-DAG:    %[[BCAST:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME:       ins(%[[ARG_SUB]] : tensor<?xf32>) outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK-SAME:       {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK:        gml_st.set_yield %[[BCAST]] into %[[INIT]][%[[BCAST_TILE]]]
  // CHECK:      return %[[RESULT]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape)
      {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @log_log_bcast
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %{{.*}}: tensor<2xindex>
func.func @log_log_bcast(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>)
    -> tensor<?x?xf32> {
  // CHECK-NOT:  linalg.generic
  // CHECK-NOT:  dynamic_broadcast_in_dim
  // CHECK:      %[[RESULT:.*]] = gml_st.parallel
  // CHECK-DAG:    %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE:29]]]
  // CHECK-DAG:    %[[INIT_GENERIC0_SUB:.*]] = gml_st.materialize %[[INIT_GENERIC0:.*]][%[[ARG_TILE]]]
  // CHECK-DAG:    %[[INIT_BCAST_SUB:.*]] = gml_st.materialize %[[INIT_BCAST:.*]][%[[BCAST_TILE:16]]]
  // CHECK:        %[[GENERIC0:.*]] = linalg.generic
  // CHECK-SAME:       ins(%[[ARG_SUB]] : tensor<?x?xf32>) outs(%[[INIT_GENERIC0_SUB]] : tensor<?x?xf32>)
  // CHECK:        %[[BCAST:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME:       ins(%[[GENERIC0]] : tensor<?x?xf32>) outs(%[[INIT_BCAST_SUB]] : tensor<?x?xf32>)
  // CHECK-SAME:       {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK:        %[[GENERIC1:.*]] = linalg.generic
  // CHECK-SAME:       ins(%[[BCAST]] : tensor<?x?xf32>) outs(%[[INIT_BCAST_SUB]] : tensor<?x?xf32>)
  // CHECK:        gml_st.set_yield %[[GENERIC1]] into %[[INIT_BCAST]][%[[BCAST_TILE]]]
  // CHECK:      return %[[RESULT]]
  %0 = mhlo.log %arg0 : tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %arg1)
      {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
      : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = mhlo.log %1 : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
