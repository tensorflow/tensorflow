// RUN: mlir-hlo-opt --split-input-file %s
// not_r_u_n:     --gml-st-pipeline="tile-sizes=64,4"
// TODO(b/249782181): Re-enable the test.
// not_r_u_n: FileCheck %s --check-prefix=TILE-CHECK

// not_r_u_n: mlir-hlo-opt --split-input-file %s \
// not_r_u_n:     --gml-st-pipeline="tile-sizes=1,1" | \
// not_r_u_n: FileCheck %s --check-prefix=POINT-CHECK

// TODO(akuegel): Also run with the option lower-to-loops. This fails currently
// due to not having a bufferization for thlo.dynamic_broadcast_in_dim.

func.func @log(%arg0: tensor<512x4xf32>) -> tensor<512x4xf32> {
  %0 = mhlo.log %arg0 : tensor<512x4xf32>
  return %0 : tensor<512x4xf32>
}

// TILE-CHECK-LABEL: @log
// TILE-CHECK-SAME:  %[[ARG:.*]]: tensor<512x4xf32>
// TILE-CHECK:       %[[RES:.*]] = gml_st.parallel
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// TILE-CHECK:         %[[LOG_SUB:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<64x4xf32>)
// TILE-CHECK:         gml_st.set_yield %[[LOG_SUB]]
// TILE-CHECK:       return %[[RES]]

// POINT-CHECK-LABEL: @log
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<512x4xf32>
// POINT-CHECK:       %[[RES:.*]] = gml_st.parallel
// POINT-CHECK-DAG:     %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// POINT-CHECK-DAG:     %[[LOG_SUB:.*]] = math.log %[[ARG_SUB]]
// POINT-CHECK:         gml_st.set_yield %[[LOG_SUB]]
// POINT-CHECK:       return %[[RES]]

// -----

func.func @transposed_log(%arg0: tensor<20x64xf32>) -> tensor<64x20xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} :
      (tensor<20x64xf32>) -> tensor<64x20xf32>
  %1 = mhlo.log %0 : tensor<64x20xf32>
  return %1 : tensor<64x20xf32>
}

// TILE-CHECK-LABEL: @transposed_log
// TILE-CHECK-SAME:  %[[ARG:.*]]: tensor<20x64xf32>
// TILE-CHECK:       %[[RES:.*]] = gml_st.parallel
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// TILE-CHECK:         %[[TRANSPOSE_SUB:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<4x64xf32>)
// TILE-CHECK:         %[[LOG_SUB:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[TRANSPOSE_SUB]] : tensor<64x4xf32>)
// TILE-CHECK:         gml_st.set_yield %[[LOG_SUB]]
// TILE-CHECK:       return %[[RES]]

// POINT-CHECK:      @transposed_log
// POINT-CHECK-SAME: %[[ARG:.*]]: tensor<20x64xf32>)
// POINT-CHECK:      %[[RES:.*]] = gml_st.parallel
// POINT-CHECK:        %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// POINT-CHECK:        %[[LOG_SUB:.*]] = math.log %[[ARG_SUB]]
// POINT-CHECK:        gml_st.set_yield %[[LOG_SUB]]
// POINT-CHECK:      return %[[RES]]

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
// TILE-CHECK:       %[[RES:.*]] = gml_st.parallel
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// TILE-CHECK:         %[[BCAST_SUB:.*]] = thlo.dynamic_broadcast_in_dim
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<?xf32>)
// TILE-CHECK:         gml_st.set_yield %[[BCAST_SUB]]
// TILE-CHECK:       return %[[RES]]

// POINT-CHECK-LABEL: @broadcast_in_dim
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// POINT-CHECK:       %[[RES:.*]] = gml_st.parallel
// POINT-CHECK:         %[[BCAST_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// POINT-CHECK:         gml_st.set_yield %[[BCAST_SUB]]
// POINT-CHECK:       return %[[RES]]

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
// TILE-CHECK:       %[[RES:.*]] = gml_st.parallel
// TILE-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// TILE-CHECK:         %[[LOG_SUB:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[ARG_SUB]] : tensor<?x?xf32>)
// TILE-CHECK:         %[[BCAST_SUB:.*]] = thlo.dynamic_broadcast_in_dim
// TILE-CHECK-SAME:        ins(%[[LOG_SUB]] : tensor<?x?xf32>)
// TILE-CHECK:         %[[LOG_LOG_SUB:.*]] = linalg.generic
// TILE-CHECK-SAME:        ins(%[[BCAST_SUB]] : tensor<?x?xf32>)
// TILE-CHECK:         gml_st.set_yield %[[LOG_LOG_SUB]]
// TILE-CHECK:       return %[[RES]]

// POINT-CHECK-LABEL: @log_log_bcast
// POINT-CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
// POINT-CHECK:       %[[RES:.*]] = gml_st.parallel
// POINT-CHECK:         %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%{{.*}}]
// POINT-CHECK:         %[[LOG_SUB:.*]] = math.log %[[ARG_SUB]]
// POINT-CHECK:         %[[LOG_LOG_SUB:.*]] = math.log %[[LOG_SUB]]
// POINT-CHECK:         gml_st.set_yield %[[LOG_LOG_SUB]]
// POINT-CHECK:       return %[[RES]]

// -----

func.func @concat(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 }
      : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %concat : tensor<?x?xf32>
}

// POINT-CHECK-LABEL: @concat
// POINT-CHECK-SAME:  %[[ARG_A:.*]]: tensor<?x?xf32>, %[[ARG_B:.*]]: tensor<?x?xf32>, %[[ARG_C:.*]]: tensor<?x?xf32>
// POINT-CHECK:       %[[RESULT:.*]] = gml_st.parallel
// POINT-CHECK:         %[[RESULT_IN_ABC:.*]] = scf.if
// POINT-CHECK:           %[[RESULT_IN_A:.*]] = gml_st.materialize %[[ARG_A]][%{{.*}}]
// POINT-CHECK:           scf.yield %[[RESULT_IN_A]]
// POINT-CHECK:         else
// POINT-CHECK:           %[[RESULT_IN_BC:.*]] = scf.if
// POINT-CHECK:             %[[RESULT_IN_B:.*]] = gml_st.materialize %[[ARG_B]][%{{.*}}]
// POINT-CHECK:             scf.yield %[[RESULT_IN_B]]
// POINT-CHECK:           else
// POINT-CHECK:             %[[RESULT_IN_C:.*]] = gml_st.materialize %[[ARG_C]][%{{.*}}]
// POINT-CHECK:             scf.yield %[[RESULT_IN_C]]
// POINT-CHECK:           scf.yield %[[RESULT_IN_BC]]
// POINT-CHECK:         gml_st.set_yield %[[RESULT_IN_ABC]]
// POINT-CHECK:       return %[[RESULT]]

// TILE-CHECK-LABEL: @concat
// TILE-CHECK-SAME:  %[[ARG_A:.*]]: tensor<?x?xf32>, %[[ARG_B:.*]]: tensor<?x?xf32>, %[[ARG_C:.*]]: tensor<?x?xf32>
// TILE-CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
// TILE-CHECK-DAG:     %[[ARG_A_SUB:.*]] = gml_st.materialize %[[ARG_A]][%{{.*}}]
// TILE-CHECK-DAG:     %[[ARG_B_SUB:.*]] = gml_st.materialize %[[ARG_B]][%{{.*}}]
// TILE-CHECK-DAG:     %[[ARG_C_SUB:.*]] = gml_st.materialize %[[ARG_C]][%{{.*}}]
// TILE-CHECK-DAG:     %[[INIT_SUB:.*]] = gml_st.materialize %{{.*}}[%{{.*}}]
// TILE-CHECK:         %[[CONCAT:.*]] = thlo.concatenate
// TILE-CHECK-SAME:        ins(%[[ARG_A_SUB]] : tensor<?x?xf32>, %[[ARG_B_SUB]] : tensor<?x?xf32>, %[[ARG_C_SUB]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        outs(%[[INIT_SUB]] : tensor<?x?xf32>)
// TILE-CHECK-SAME:        {dimension = 1 : i64}
// TILE-CHECK:         gml_st.set_yield %[[CONCAT]]
// TILE-CHECK:       return %[[PARALLEL]]
