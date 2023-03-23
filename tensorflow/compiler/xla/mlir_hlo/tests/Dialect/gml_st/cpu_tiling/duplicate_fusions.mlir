// RUN: mlir-hlo-opt %s \
// RUN:     --gml-st-cpu-tiling-pipeline="enable-fusion-clusters=true enable-fusion-cluster-outlining=true" | \
// RUN: FileCheck %s

func.func @double_bcast_map_reduce(%arg : tensor<?xf32>,
    %init_3d : tensor<?x?x?xf32>, %init_1d : tensor<?xf32>) -> tensor<?xf32> {

  // Bcast, map, reduce.
  %broadcasted = linalg.broadcast ins(%arg : tensor<?xf32>)
      outs(%init_3d : tensor<?x?x?xf32>) dimensions = [1, 2]
  %mapped = linalg.map { math.absf } ins(%broadcasted : tensor<?x?x?xf32>)
      outs(%init_3d : tensor<?x?x?xf32>)
  %reduced = linalg.reduce { arith.addf } ins(%mapped : tensor<?x?x?xf32>)
      outs(%init_1d : tensor<?xf32>) dimensions = [1, 2]

  // And again...
  %broadcasted_ = linalg.broadcast ins(%reduced : tensor<?xf32>)
      outs(%init_3d : tensor<?x?x?xf32>) dimensions = [1, 2]
  %mapped_ = linalg.map { math.absf } ins(%broadcasted_ : tensor<?x?x?xf32>)
      outs(%init_3d : tensor<?x?x?xf32>)
  %reduced_ = linalg.reduce { arith.addf } ins(%mapped_ : tensor<?x?x?xf32>)
      outs(%init_1d : tensor<?xf32>) dimensions = [1, 2]

  return %reduced_ : tensor<?xf32>
}

// CHECK:      @[[UNIQUE_OUTLINED_FUSION_FUNC:double_bcast_map_reduce_fusion(_[0-9]+)?]]
// CHECK-SAME: %{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>
// CHECK-SAME: attributes {fusion}

// CHECK:      @double_bcast_map_reduce
// CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>, %[[INIT_3D:.*]]: tensor<?x?x?xf32>, %[[INIT_1D:.*]]: tensor<?xf32>
// CHECK:      %[[CALL_0:.*]] = call @[[UNIQUE_OUTLINED_FUSION_FUNC]](%[[INIT_3D]], %[[ARG]], %[[INIT_1D]])
// CHECK:      %[[CALL_1:.*]] = call @[[UNIQUE_OUTLINED_FUSION_FUNC]](%[[INIT_3D]], %[[CALL_0]], %[[INIT_1D]])
// CHECK:      return %[[CALL_1]]
