// RUN: mlir-hlo-opt %s --gml-st-cpu-fusion-planning \
// RUN: --split-input-file \
// RUN: | FileCheck %s

func.func @reverse_reduce_map(%input: tensor<?x?xf32>, %init0: tensor<?x?xf32>,
                              %init1: tensor<?xf32>) -> tensor<?xf32> {
  %sorted = thlo.sort
                ins(%input: tensor<?x?xf32>)
                outs(%init0: tensor<?x?xf32>)
                dimension = 0
                is_stable = false
                (%lhs: f32, %rhs: f32) {
                  %gt = arith.cmpf ogt, %lhs, %rhs: f32
                  thlo.yield %gt : i1
                }
  %reduced = linalg.reduce { arith.addf }
               ins(%sorted: tensor<?x?xf32>)
               outs(%init1: tensor<?xf32>)
               dimensions = [0]
  %result = linalg.map { math.exp }
              ins(%reduced: tensor<?xf32>)
              outs(%init1: tensor<?xf32>)
  func.return %result : tensor<?xf32>
}

// CHECK-LABEL: @reverse_reduce_map
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?xf32>, %[[INIT0:.*]]: tensor<?x?xf32>
// CHECK-SAME: %[[INIT1:.*]]: tensor

// CHECK:       %[[FUSION0:.*]] = gml_st.fusion
// CHECK-SAME:      (%[[BB_INPUT:.*]] = %[[INPUT]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[BB_INIT0:.*]] = %[[INIT0]]: tensor<?x?xf32>
// CHECK-NEXT:    %[[SORTED:.*]] = thlo.sort
// CHECK-SAME:      ins(%[[BB_INPUT]]
// CHECK-SAME:      outs(%[[BB_INIT0]]
// CHECK:         gml_st.yield %[[SORTED]]

// CHECK:       %[[FUSION1:.*]] = gml_st.fusion
// CHECK-SAME:      (%[[BB_INIT1:.*]] = %[[INIT1]]: tensor<?xf32>,
// CHECK-SAME:      %[[BB_INPUT:.*]] = %[[FUSION0]]: tensor<?x?xf32>
// CHECK:         %[[REDUCED:.*]] = linalg.reduce
// CHECK-SAME:      ins(%[[BB_INPUT]]
// CHECK-SAME:      outs(%[[BB_INIT1]]
// CHECK:         %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:      ins(%[[REDUCED]]
// CHECK-SAME:      outs(%[[BB_INIT1]]
// CHECK:         gml_st.yield %[[MAPPED]]

// CHECK:       return %[[FUSION1]]

// -----

func.func @scatter(%indices: tensor<1x1xindex>,
                           %updates: tensor<1x1x3x4xi64>,
                           %init: tensor<3x3x4xi64>) -> tensor<3x3x4xi64> {
  %res = thlo.scatter ins(%indices : tensor<1x1xindex>,
                          %updates : tensor<1x1x3x4xi64>)
                      outs(%init : tensor<3x3x4xi64>)
    (%arg5: i64, %arg6: i64) {
      thlo.yield %arg5 : i64
    }
  func.return %res : tensor<3x3x4xi64>
}

// CHECK-LABEL: func @scatter
// CHECK:       gml_st.fusion
// CHECK:         thlo.scatter
// CHECK:         gml_st.yield

// -----

func.func @sort(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                -> tensor<?x?xf32> {
  %res = thlo.sort
           ins(%input: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
           dimension = 0
           is_stable = true
           (%lhs: f32, %rhs: f32) {
             %0 = arith.cmpf ogt, %lhs, %rhs : f32
             thlo.yield %0 : i1
           }
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: func @sort
// CHECK:       gml_st.fusion
// CHECK:         thlo.sort
// CHECK:         gml_st.yield

// -----

func.func @reverse(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                   -> tensor<?x?xf32> {
  %res = thlo.reverse
           ins(%input: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
           reverse_dimensions = [0, 1]
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: func @reverse
// CHECK:       gml_st.fusion
// CHECK:         thlo.reverse
// CHECK:         gml_st.yield

// -----

func.func @transpose(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                   -> tensor<?x?xf32> {
  %res = linalg.transpose
           ins(%input: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
           permutation = [1, 0]
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: func @transpose
// CHECK:       gml_st.fusion
// CHECK:         linalg.transpose
// CHECK:         gml_st.yield

// -----

func.func @map(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                     -> tensor<?x?xf32> {
  %abs = linalg.map { math.absf } ins(%input:tensor<?x?xf32>) outs(%init:tensor<?x?xf32>)
  func.return %abs : tensor<?x?xf32>
}

// CHECK-LABEL: func @map
// CHECK:       gml_st.fusion
// CHECK:         linalg.map
// CHECK:         gml_st.yield

// -----

func.func @map_non_unique_users(%arg: tensor<?x?xf32>,
                                %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %exp = linalg.map { math.exp }
           ins(%arg: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
  %mul = linalg.map { arith.mulf }
           ins(%exp, %exp: tensor<?x?xf32>, tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
  %abs = linalg.map { math.absf }
           ins(%mul: tensor<?x?xf32>)
           outs(%init: tensor<?x?xf32>)
  func.return %abs : tensor<?x?xf32>
}

// CHECK-LABEL:   func @map_non_unique_users
// CHECK:         gml_st.fusion
// CHECK-COUNT-3:   linalg.map
// CHECK:           gml_st.yield

// -----

func.func @matmul(%input1: tensor<4x8xf32>, %input2: tensor<8x16xf32>,
                  %init: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %res = linalg.matmul
           ins(%input1, %input2 : tensor<4x8xf32>, tensor<8x16xf32>)
           outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  func.return %res : tensor<4x16xf32>
}

// CHECK-LABEL: func @matmul
// CHECK:       gml_st.fusion
// CHECK:         linalg.matmul
// CHECK:         gml_st.yield

// -----

func.func @reduce(%input: tensor<100x10xf32>,
                        %output: tensor<10xf32>) -> tensor<10xf32> {
  %res = linalg.reduce { arith.addf }
           ins(%input: tensor<100x10xf32>)
           outs(%output: tensor<10xf32>)
           dimensions = [0]
  return %res : tensor<10xf32>
}

// CHECK-LABEL:   func @reduce
// CHECK:         gml_st.fusion
// CHECK:           linalg.reduce
// CHECK:           gml_st.yield

// -----

func.func @fused_matmul(%arg0: tensor<1x32xf32>, %arg1: tensor<32x10xf32>,
                        %arg2: tensor<10xf32>) -> tensor<1x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x10xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x10xf32>) -> tensor<1x10xf32>
  %2 = linalg.matmul
         ins(%arg0, %arg1 : tensor<1x32xf32>, tensor<32x10xf32>)
         outs(%1 : tensor<1x10xf32>) -> tensor<1x10xf32>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<10xf32> into tensor<1x10xf32>
  %mapped = linalg.map { arith.addf }
              ins(%2, %expanded : tensor<1x10xf32>, tensor<1x10xf32>)
              outs(%0 : tensor<1x10xf32>)
  return %mapped : tensor<1x10xf32>
}

// CHECK-LABEL: func @fused_matmul
// CHECK-SAME:      (%[[ARG0:.*]]: tensor<1x32xf32>, %[[ARG1:.*]]: tensor<32x10xf32>
// CHECK-SAME:      %[[ARG2:.*]]: tensor<10xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0
// CHECK:         %[[EMPTY:.*]] = tensor.empty()
// CHECK:         gml_st.fusion
// CHECK-SAME:        (%[[EMPTY_:.*]] = %[[EMPTY]]: tensor<1x10xf32>,
// CHECK-SAME:        %[[ARG2_:.*]] = %[[ARG2]]: tensor<10xf32>
// CHECK-SAME:        %[[ARG0_:.*]] = %[[ARG0]]: tensor<1x32xf32>
// CHECK-SAME:        %[[ARG1_:.*]] = %[[ARG1]]: tensor<32x10xf32>
// CHECK-SAME:        %[[C0_:.*]] = %[[C0]]: f32
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG2_]]
// CHECK:           %[[FILLED:.*]] = linalg.fill
// CHECK-SAME:        ins(%[[C0_]] : f32)
// CHECK-SAME:        outs(%[[EMPTY_]] : tensor<1x10xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:        ins(%[[ARG0_]], %[[ARG1_]]
// CHECK-SAME:        outs(%[[FILLED]]
// CHECK:           %[[MAP:.*]] = linalg.map
// CHECK-SAME:        ins(%[[MATMUL]], %[[EXPANDED]]
// CHECK-SAME:        outs(%[[EMPTY_]]
// CHECK:           gml_st.yield %[[MAP]]
