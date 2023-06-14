// RUN: mlir-hlo-opt %s \
// RUN:     --gml-fusion-outlining --duplicate-function-elimination | \
// RUN: FileCheck %s

func.func @double_bcast_map_reduce(%arg : tensor<?xf32>,
    %init_3d : tensor<?x?x?xf32>, %init_1d : tensor<?xf32>) -> tensor<?xf32> {

  // Bcast, map, reduce.
  %0 = gml_st.fusion ins(%arg_ = %arg : tensor<?xf32>,
                         %init_3d_ = %init_3d : tensor<?x?x?xf32>)
                     inits(%init_1d_ = %init_1d : tensor<?xf32>) {
    %broadcasted = linalg.broadcast ins(%arg_ : tensor<?xf32>)
        outs(%init_3d_ : tensor<?x?x?xf32>) dimensions = [1, 2]
    %mapped = linalg.map { math.absf } ins(%broadcasted : tensor<?x?x?xf32>)
        outs(%init_3d_ : tensor<?x?x?xf32>)
    %reduced = linalg.reduce { arith.addf } ins(%mapped : tensor<?x?x?xf32>)
        outs(%init_1d_ : tensor<?xf32>) dimensions = [1, 2]
    gml_st.yield %reduced : tensor<?xf32>
  } : tensor<?xf32>

  // And again...
  %1 = gml_st.fusion ins(%arg_ = %0 : tensor<?xf32>,
                         %init_3d_ = %init_3d : tensor<?x?x?xf32>)
                     inits(%init_1d_ = %init_1d : tensor<?xf32>) {
    %broadcasted = linalg.broadcast ins(%arg_ : tensor<?xf32>)
        outs(%init_3d_ : tensor<?x?x?xf32>) dimensions = [1, 2]
    %mapped = linalg.map { math.absf } ins(%broadcasted : tensor<?x?x?xf32>)
        outs(%init_3d_ : tensor<?x?x?xf32>)
    %reduced = linalg.reduce { arith.addf } ins(%mapped : tensor<?x?x?xf32>)
        outs(%init_1d_ : tensor<?xf32>) dimensions = [1, 2]
    gml_st.yield %reduced : tensor<?xf32>
  } : tensor<?xf32>

  return %1 : tensor<?xf32>
}

// CHECK:      @[[UNIQUE_OUTLINED_FUSION_FUNC:double_bcast_map_reduce_fusion(_[0-9]+)?]]
// CHECK-SAME: %{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xf32>
// CHECK-SAME: attributes {fusion}

// CHECK:      @double_bcast_map_reduce
// CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>, %[[INIT_3D:.*]]: tensor<?x?x?xf32>, %[[INIT_1D:.*]]: tensor<?xf32>
// CHECK:      %[[CALL_0:.*]] = call @[[UNIQUE_OUTLINED_FUSION_FUNC]](%[[ARG]], %[[INIT_3D]], %[[INIT_1D]])
// CHECK:      %[[CALL_1:.*]] = call @[[UNIQUE_OUTLINED_FUSION_FUNC]](%[[CALL_0]], %[[INIT_3D]], %[[INIT_1D]])
// CHECK:      return %[[CALL_1]]
