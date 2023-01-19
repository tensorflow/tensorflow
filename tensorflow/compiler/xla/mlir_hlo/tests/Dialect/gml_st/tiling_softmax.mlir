// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling-softmax="tile-sizes=8,16 distribute=true distribution-label=test" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

// CHECK-LABEL: @partial_softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @partial_softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:   %[[C64:.*]] = arith.constant 64
  // CHECK-DAG:   %[[CST:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:      (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C8]])
  // CHECK:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]] [%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE:.*]] = linalg.reduce
  // CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         (%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32) {
  // CHECK:           %[[MAXF:.*]] = arith.maxf %[[ARG3]], %[[ARG4]]
  // CHECK:           linalg.yield %[[MAXF]]
  // CHECK:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[BROADCAST:.*]] = linalg.broadcast
  // CHECK-SAME:        ins(%[[REDUCE]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[MAP:.*]] = linalg.map { arith.subf }
  // CHECK-SAME:        ins(%[[MATERIALIZE]], %[[BROADCAST]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)

  // CHECK:       return %[[PARALLEL]]
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.reduce ins(%arg0 : tensor<64x128xf32>)
                     outs(%1 : tensor<64xf32>)
                     dimensions = [1]
    (%arg1: f32, %arg2: f32) {
      %6 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %6 : f32
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
  // CHECK-DAG:   %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK-DAG:   %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]] [%[[ARG1]]] [8] [1]
  // CHECK:       %[[REDUCE:.*]] = linalg.reduce { arith.maxf }
  // CHECK-SAME:      ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:      outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:      dimensions = [1]
  // CHECK-DAG:   %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]] [%[[ARG1]], 0] [8, 128] [1, 1]
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
  %2 = linalg.reduce
      ins(%arg0 : tensor<64x128xf32>)
      outs(%1 : tensor<64xf32>)
      dimensions = [1]
  (%arg2: f32, %arg3: f32) {
    %9 = arith.maxf %arg3, %arg2 : f32
    linalg.yield %9 : f32
  }
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.broadcast
      ins(%2 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>)
      dimensions = [1]
  %5 = linalg.map
      ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>)
  (%arg2: f32, %arg3: f32) {
    %9 = arith.subf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  }
  %8 = gml_st.materialize %5[%arg1, 0] [8, 128] [1, 1]
      : tensor<64x128xf32> to tensor<8x128xf32>
  return %8 : tensor<8x128xf32>
}

// -----

// CHECK-LABEL: @softmax
// CHECK-SAME:  %[[ARG0:.*]]: tensor<64x128xf32>
func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:   %[[C64:.*]] = arith.constant 64
  // CHECK-DAG:   %[[CST:.*]] = arith.constant -0.000000e+00
  // CHECK-DAG:   %[[CST_0:.*]] = arith.constant 0xFF800000
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK-DAG:   %[[INIT_0:.*]] = tensor.empty() : tensor<64x128xf32>
  // CHECK-DAG:   %[[FILL_0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<64xf32>)
  // CHECK:       %[[PARALLEL:.*]] = gml_st.parallel
  // CHECK-SAME:      (%[[ARG1:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C8]])
  // CHECK:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]] [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[FILL]] [%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE:.*]] = linalg.reduce
  // CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_0]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         (%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32) {
  // CHECK:           %[[MAXF:.*]] = arith.maxf %[[ARG3]], %[[ARG4]]
  // CHECK:           linalg.yield %[[MAXF]]
  // CHECK:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT_0]] [%[[ARG1]], 0] [8, 128] [1, 1]
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
  // CHECK:         %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[FILL_0]] [%[[ARG1]]] [8] [1]
  // CHECK:         %[[REDUCE_0:.*]] = linalg.reduce { arith.addf }
  // CHECK-SAME:        ins(%[[MAP_0]] : tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_3]] : tensor<8xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[BROADCAST_0:.*]] = linalg.broadcast
  // CHECK-SAME:        ins(%[[REDUCE_0]] : tensor<8xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK-SAME:        dimensions = [1]
  // CHECK:         %[[MAP_1:.*]] = linalg.map { arith.divf }
  // CHECK-SAME:        ins(%[[MAP_0]], %[[BROADCAST_0]] : tensor<8x128xf32>, tensor<8x128xf32>)
  // CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x128xf32>)
  // CHECK:         %[[TILE:.*]] = gml_st.tile [%[[ARG1]], 0] [8, 128] [1, 1]
  // CHECK:         gml_st.set_yield %[[MAP_1]] into %[[INIT_0]][%[[TILE]]]
  // CHECK:       return %[[PARALLEL]]
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
