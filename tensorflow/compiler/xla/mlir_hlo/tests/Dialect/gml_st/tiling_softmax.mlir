// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling-softmax="tile-sizes=8,16" --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK-LABEL: @partial_softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @partial_softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK:       %[[CST:.*]] = arith.constant 0xFF800000
  // CHECK:       %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK:       %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK:       %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK:       %[[PARALLEL:.*]] = scf.forall
  // CHECK-SAME:      (%[[ARG1:.*]]) = (0) to (64) step (8)
  // CHECK-SAME:      shared_outs(%[[INIT_0_:.*]] = %[[INIT_0]])
  // CHECK:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[FILL]][%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  // CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[INIT_0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[BROADCAST:.*]] = linalg.broadcast
  // CHECK-SAME:        ins(%[[REDUCE]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[INIT_0_SUB:.*]] = tensor.extract_slice %[[INIT_0_]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MAP:.*]] = linalg.map { arith.subf }
  // CHECK-SAME:        ins(%[[MATERIALIZE]], %[[BROADCAST]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[INIT_0_SUB]] : tensor<8x128xf32>)

  // CHECK:       return %[[PARALLEL]]
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.reduce { arith.maxf }
         ins(%arg0 : tensor<64x128xf32>)
         outs(%1 : tensor<64xf32>)
         dimensions = [1]
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.broadcast
         ins(%2 : tensor<64xf32>)
         outs(%3 : tensor<64x128xf32>)
         dimensions = [1]
  %5 = linalg.map { arith.subf }
         ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
         outs(%3 : tensor<64x128xf32>)
  return %5 : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: @partial_softmax_fusion
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>, %[[ARG1:.*]]: index
func.func @partial_softmax_fusion(%arg0: tensor<64x128xf32>, %arg1: index)
    -> tensor<8x128xf32> {
  // CHECK-DAG:   %[[CST:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK-DAG:   %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK-DAG:   %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[FILL]][%[[ARG1]]] [8] [1]
  // CHECK:       %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  // CHECK-SAME:      ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:      outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:      dimensions = [1]
  // CHECK-DAG:   %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[INIT_0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:       %[[BROADCAST:.*]] = linalg.broadcast
  // CHECK-SAME:      ins(%[[REDUCE]] : tensor<8xf32>)
  // CHECK-SAME:      outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:      dimensions = [1]
  // CHECK:       %[[MAP:.*]] = linalg.map { arith.subf }
  // CHECK-SAME:      ins(%[[MATERIALIZE]], %[[BROADCAST]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:      outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:       return %[[MAP]]
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.reduce { arith.maxf }
         ins(%arg0 : tensor<64x128xf32>)
         outs(%1 : tensor<64xf32>)
         dimensions = [1]
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.broadcast
         ins(%2 : tensor<64xf32>)
         outs(%3 : tensor<64x128xf32>)
         dimensions = [1]
  %5 = linalg.map { arith.subf }
         ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
         outs(%3 : tensor<64x128xf32>)
  %8 = tensor.extract_slice %5[%arg1, 0] [8, 128] [1, 1]
      : tensor<64x128xf32> to tensor<8x128xf32>
  return %8 : tensor<8x128xf32>
}

// -----

// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-DAG:   %[[CST:.*]] = arith.constant -0.000000e+00
  // CHECK-DAG:   %[[CST_0:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK-DAG:   %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK:       %[[PARALLEL:.*]] = scf.forall
  // CHECK-SAME:      (%[[ARG1:.*]]) = (0) to (64) step (8)
  // CHECK-SAME:      shared_outs(%[[INIT_0_:.*]] = %[[INIT_0]])
  // CHECK:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[FILL]][%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  // CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[INIT_0]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[BROADCAST:.*]] = linalg.broadcast
  // CHECK-SAME:        ins(%[[REDUCE]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[MAP:.*]] = linalg.map { arith.subf }
  // CHECK-SAME:        ins(%[[MATERIALIZE]], %[[BROADCAST]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:         %[[MAP_0:.*]] = linalg.map { math.exp }
  // CHECK-SAME:        ins(%[[MAP]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:         %[[MATERIALIZE_3:.*]] = tensor.extract_slice %[[FILL_0]][%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE_0:.*]] = linalg.reduce { arith.addf }
  // CHECK-SAME:        ins(%[[MAP_0]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_3]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[BROADCAST_0:.*]] = linalg.broadcast
  // CHECK-SAME:        ins(%[[REDUCE_0]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[INIT_0_SUB:.*]] = tensor.extract_slice %[[INIT_0_]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MAP_1:.*]] = linalg.map { arith.divf }
  // CHECK-SAME:        ins(%[[MAP_0]], %[[BROADCAST_0]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[INIT_0_SUB]] : tensor<8x128xf32>)
  // CHECK:         tensor.parallel_insert_slice %[[MAP_1]] into %[[INIT_0_]][%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:       return %[[PARALLEL]]
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.reduce { arith.maxf }
         ins(%arg0 : tensor<64x128xf32>)
         outs(%1 : tensor<64xf32>) dimensions = [1]
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.broadcast
         ins(%2 : tensor<64xf32>)
         outs(%3 : tensor<64x128xf32>)
         dimensions = [1]
  %5 = linalg.map { arith.subf }
         ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
         outs(%3 : tensor<64x128xf32>)
  %6 = linalg.map { math.exp }
         ins(%5 : tensor<64x128xf32>)
         outs(%3 : tensor<64x128xf32>)
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %8 = linalg.reduce { arith.addf }
         ins(%6 : tensor<64x128xf32>)
         outs(%7 : tensor<64xf32>)
         dimensions = [1]
  %9 = linalg.broadcast
         ins(%8 : tensor<64xf32>)
         outs(%3 : tensor<64x128xf32>)
         dimensions = [1]
  %10 = linalg.map { arith.divf }
          ins(%6, %9 : tensor<64x128xf32>, tensor<64x128xf32>)
          outs(%3 : tensor<64x128xf32>)
  return %10 : tensor<64x128xf32>
}
