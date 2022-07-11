// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:  --gml-st-pipeline="tile-sizes=64,4 fuse" \
// RUN: | FileCheck %s

// TODO(akuegel): Also run with the option lower-to-loops. This fails currently
// due to not having a bufferization for gml_st.dynamic_broadcast_in_dim.

// CHECK-LABEL: func @log(
// CHECK-SAME: %[[ARG0:.*]]: tensor<512x4xf32>)
func.func @log(%arg0: tensor<512x4xf32>) -> tensor<512x4xf32> {
  %0 = mhlo.log %arg0 : tensor<512x4xf32>
  return %0 : tensor<512x4xf32>
}
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK:     %[[INIT:.*]] = linalg.init_tensor [512, 4] : tensor<512x4xf32>
// CHECK:     %[[SPACE:.*]] = gml_st.space [512, 4] : !gml_st.tile<512x4>
// CHECK:     %[[RESULT:.*]] = gml_st.parallel (%[[IV:.*]], %[[IV2:.*]]) =
// CHECK:         (%[[C0]], %[[C0]]) to (%[[C512]], %[[C4]])
// CHECK:         step (%[[C64]], %[[C4]])
// CHECK:       %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]]
// CHECK:                                             [64, 4] [1, 1] :
// CHECK:                      !gml_st.tile<512x4> to !gml_st.tile<64x4>
// CHECK:       %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]] :
// CHECK:                         tensor<512x4xf32>[!gml_st.tile<64x4>]
// CHECK:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]] :
// CHECK:                          tensor<512x4xf32>[!gml_st.tile<64x4>]
// CHECK:       %[[LINALG_OP:.*]] = linalg.generic
// CHECK:                           ins(%[[ARG_SUB]] : tensor<64x4xf32>)
// CHECK:                           outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
// CHECK:         %[[LOG:.*]] = math.log %{{.*}} : f32
// CHECK:         linalg.yield %[[LOG]] : f32
// CHECK:       gml_st.set_yield %[[LINALG_OP]] into %[[INIT]][%[[TILE]]] :
// CHECK:           tensor<64x4xf32> into tensor<512x4xf32>[!gml_st.tile<64x4>]
// CHECK:     return %[[RESULT]] : tensor<512x4xf32>

// -----

// CHECK-LABEL: func @transposed_log(
// CHECK-SAME: %[[ARG0:.*]]: tensor<20x64xf32>)
func.func @transposed_log(%arg0: tensor<20x64xf32>) -> tensor<64x20xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} :
      (tensor<20x64xf32>) -> tensor<64x20xf32>
  %1 = mhlo.log %0 : tensor<64x20xf32>
  return %1 : tensor<64x20xf32>
}
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK:     %[[INIT:.*]] = linalg.init_tensor [64, 20] : tensor<64x20xf32>
// CHECK:     %[[INIT2:.*]] = linalg.init_tensor [64, 20] : tensor<64x20xf32>
// CHECK:     %[[SPACE:.*]] = gml_st.space [64, 20] : !gml_st.tile<64x20>
// CHECK:     %[[RESULT:.*]] = gml_st.parallel (%[[IV:.*]], %[[IV2:.*]]) =
// CHECK:         (%[[C0]], %[[C0]]) to (%[[C64]], %[[C20]])
// CHECK:         step (%[[C64]], %[[C4]])
// CHECK:       %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]], %[[IV2]]]
// CHECK:                                             [64, 4] [1, 1] :
// CHECK:                      !gml_st.tile<64x20> to !gml_st.tile<64x4>
// CHECK:       %[[SPACE2:.*]] = gml_st.space [20, 64] : !gml_st.tile<20x64>
// CHECK:       %[[TILE2:.*]] = gml_st.tile %[[SPACE2]] [%[[IV2]], %[[IV]]]
// CHECK:                                               [4, 64] [1, 1] :
// CHECK:                       !gml_st.tile<20x64> to !gml_st.tile<4x64>
// CHECK:       %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]][%[[TILE2]]] :
// CHECK:                         tensor<20x64xf32>[!gml_st.tile<4x64>]
// CHECK:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]] :
// CHECK:                          tensor<64x20xf32>[!gml_st.tile<64x4>]
// CHECK:       %[[LINALG_OP:.*]] = linalg.generic
// CHECK:                           ins(%[[ARG_SUB]] : tensor<4x64xf32>)
// CHECK:                           outs(%[[INIT_SUB:.*]] : tensor<64x4xf32>)
// CHECK:       %[[TRANSPOSE_SUB:.*]] = gml_st.materialize %[[INIT2]][%[[TILE]]]
// CHECK:                               : tensor<64x20xf32>[!gml_st.tile<64x4>]
// CHECK:       %[[LOG_RES:.*]] = linalg.generic
// CHECK:                         ins(%[[LINALG_OP]] : tensor<64x4xf32>)
// CHECK:                         outs(%[[TRANSPOSE_SUB:.*]] : tensor<64x4xf32>)
// CHECK:         %[[LOG:.*]] = math.log %{{.*}} : f32
// CHECK:         linalg.yield %[[LOG]] : f32
// CHECK:       gml_st.set_yield %[[LOG_RES]] into %[[INIT2]][%[[TILE]]] :
// CHECK:           tensor<64x4xf32> into tensor<64x20xf32>[!gml_st.tile<64x4>]
// CHECK:     return %[[RESULT]] : tensor<64x20xf32>

// -----

// CHECK-LABEL: func @broadcast_in_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[SHAPE:.*]]: tensor<2xindex>)
func.func @broadcast_in_dim(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape)
      {broadcast_dimensions = dense<[1]> : tensor<1xi64>} :
      (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK: gml_st.dynamic_broadcast_in_dim
