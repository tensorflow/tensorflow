// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling-softmax="tile-sizes=8,16" --canonicalize --cse \
// RUN:     --gml-tiling-softmax="tile-sizes=1,1" --canonicalize --cse | \
// RUN: FileCheck %s

func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.reduce ins(%arg0 : tensor<64x128xf32>)
                     outs(%1 : tensor<64xf32>) dimensions = [1]
    (%arg1: f32, %arg2: f32) {
      %11 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    }
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.broadcast
    ins(%2 : tensor<64xf32>)
    outs(%3 : tensor<64x128xf32>)
    dimensions = [1]
  %5 = linalg.map ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
              outs(%3 : tensor<64x128xf32>)
  (%arg1: f32, %arg2: f32) {
    %11 = arith.subf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  }
  %6 = linalg.map ins(%5 : tensor<64x128xf32>)
              outs(%3 : tensor<64x128xf32>)
  (%arg1: f32) {
    %11 = math.exp %arg1 : f32
    linalg.yield %11 : f32
  }
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %8 = linalg.reduce ins(%6 : tensor<64x128xf32>)
                     outs(%7 : tensor<64xf32>) dimensions = [1]
    (%arg1: f32, %arg2: f32) {
      %11 = arith.addf %arg2, %arg1 : f32
      linalg.yield %11 : f32
    }
  %9 = linalg.broadcast
    ins(%8 : tensor<64xf32>)
    outs(%3 : tensor<64x128xf32>)
    dimensions = [1]
  %10 = linalg.map ins(%6, %9 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>)
  (%arg1: f32, %arg2: f32) {
    %11 = arith.divf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  }
  return %10 : tensor<64x128xf32>
}
// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
// CHECK-DAG:  %[[CST:.*]] = arith.constant -0.000000e+00
// CHECK-DAG:  %[[CST_0:.*]] = arith.constant 0xFF800000
// CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32)
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)
// CHECK:      %[[EMPTY_0:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK:      %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32)
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<64xf32>)

// CHECK:      %[[PARALLEL:.*]] = scf.forall (%[[ARG1:.*]]) = (0) to (64) step (8)
// CHECK-SAME:     shared_outs(%[[EMPTY_:.*]] = %[[EMPTY_0]])
// CHECK-DAG:   %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0] [8, 128] [1, 1]
// CHECK-DAG:   %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[FILL]][%[[ARG1]]] [8] [1]
// CHECK-DAG:   %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[EMPTY_0]][%[[ARG1]], 0] [8, 128] [1, 1]
// CHECK-DAG:   %[[MATERIALIZE_3:.*]] = tensor.extract_slice %[[FILL_0]][%[[ARG1]]] [8] [1]
// CHECK-DAG:   %[[EMPTY_SUB:.*]] = tensor.extract_slice %[[EMPTY_]]

// CHECK:        %[[PARALLEL_0:.*]] = scf.forall (%[[ARG2:.*]]) in (8)
// CHECK-SAME:       shared_outs(%[[EMPTY_SUB_:.*]] = %[[EMPTY_SUB]])
// CHECK-NEXT:     %[[MATERIALIZE_4:.*]] = tensor.extract_slice %[[MATERIALIZE]][%[[ARG2]], 0] [1, 128] [1, 1]
// CHECK-NEXT:     %[[MATERIALIZE_5:.*]] = tensor.extract_slice %[[MATERIALIZE_0]][%[[ARG2]]] [1] [1]
// CHECK-NEXT:     %[[REDUCE:.*]] = linalg.reduce
// CHECK-SAME:         ins(%[[MATERIALIZE_4]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_5]] : tensor<1xf32>)
// CHECK-SAME:         dimensions = [1]

// CHECK:          %[[MATERIALIZE_6:.*]] = tensor.extract_slice %[[MATERIALIZE_1]][%[[ARG2]], 0] [1, 128] [1, 1]
// CHECK-NEXT:     %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-SAME:         ins(%[[REDUCE]] : tensor<1xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)
// CHECK-SAME:         dimensions = [1]

// CHECK:          %[[MAP:.*]] = linalg.map
// CHECK-SAME:         ins(%[[MATERIALIZE_4]], %[[BROADCAST]] : tensor<1x128xf32>, tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[MAP_0:.*]] = linalg.map
// CHECK-SAME:         ins(%[[MAP]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK:          %[[MATERIALIZE_8:.*]] = tensor.extract_slice %[[MATERIALIZE_3]][%[[ARG2]]] [1] [1]
// CHECK-NEXT:          %[[REDUCE_0:.*]] = linalg.reduce
// CHECK-SAME:         ins(%[[MAP_0]] : tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_8]] : tensor<1xf32>)

// CHECK:          %[[BROADCAST_0:.*]] = linalg.broadcast
// CHECK-SAME:         ins(%[[REDUCE_0]] : tensor<1xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_6]] : tensor<1x128xf32>)

// CHECK-NEXT:     %[[MATERIALIZE_7:.*]] = tensor.extract_slice %[[EMPTY_SUB_]]
// CHECK:          %[[MAP_1:.*]] = linalg.map
// CHECK-SAME:         ins(%[[MAP_0]], %[[BROADCAST_0]] : tensor<1x128xf32>, tensor<1x128xf32>)
// CHECK-SAME:         outs(%[[MATERIALIZE_7]] : tensor<1x128xf32>)
// CHECK:          tensor.parallel_insert_slice %[[MAP_1]] into %[[EMPTY_SUB_]][%[[ARG2]], 0] [1, 128] [1, 1]
// CHECK:        tensor.parallel_insert_slice %[[PARALLEL_0]] into %[[EMPTY_]][%[[ARG1]], 0] [8, 128] [1, 1]
// CHECK:      return %[[PARALLEL]]
