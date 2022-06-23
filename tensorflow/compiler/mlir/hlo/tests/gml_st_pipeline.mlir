// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:  --gml-st-pipeline="tile-sizes=256" \
// RUN: | FileCheck --dump-input=always %s

// CHECK-LABEL: func @log(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2048xf32>)
func.func @log(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
  %0 = mhlo.log %arg0 : tensor<2048xf32>
  return %0 : tensor<2048xf32>
}
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
// CHECK:     %[[INIT:.*]] = linalg.init_tensor [2048] : tensor<2048xf32>
// CHECK:     %[[SPACE:.*]] = gml_st.space [2048] : !gml_st.tile<2048>
// CHECK:     %[[RESULT:.*]] = gml_st.parallel (%[[IV:.*]]) = (%[[C0]])
// CHECK:         to (%[[C2048]]) step (%[[C256]])
// CHECK:         outs (%[[INIT]]: tensor<2048xf32>)
// CHECK:       %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]]] [256] [1] :
// CHECK:                      !gml_st.tile<2048> to !gml_st.tile<256>
// CHECK:       %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG0]] at %[[TILE]] :
// CHECK:                         tensor<2048xf32> at !gml_st.tile<256>
// CHECK:       %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]] at %[[TILE]] :
// CHECK:                          tensor<2048xf32> at !gml_st.tile<256>
// CHECK:       %[[LINALG_OP:.*]] = linalg.generic
// CHECK:                           ins(%[[ARG_SUB]] : tensor<256xf32>)
// CHECK:                           outs(%[[INIT_SUB:.*]] : tensor<256xf32>)
// CHECK:         %[[LOG:.*]] = math.log %{{.*}} : f32
// CHECK:         linalg.yield %[[LOG]] : f32
// CHECK:       gml_st.subset_yield %[[LINALG_OP]] at %[[TILE]] :
// CHECK:           tensor<256xf32> at !gml_st.tile<256>
// CHECK:     return %[[RESULT]] : tensor<2048xf32>

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
